//===--------- SARIFDiagnostic.cpp - SARIF Diagnostic Formatting ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/SARIFDiagnostic.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Locale.h"
#include <algorithm>
#include <string>

namespace clang {

// In sarif mode,
// a diagnostics 'group' have 1 top-level error/warning and several sub-level
// notes. For example:
//
// error: static assertion failed.
//   note: in instantiation of 'cat::meow'.
//     note: because concept 'paper_tiger' would be invalid.
// error: invalid operands to binary expression 'cat::meow' and 'dog::wolf'.
//   note: candidate function not viable.
//     note: no known conversion from 'tiger::meooooow' to 'cat::meow'
//   note: candidate function ignored.
//     note: constraints not satisfied.
//   note: ... (candidates)
//     note: ... (reasons)
//   note: too many candidates.
// error: too many errors occured, stopping now.

SARIFDiagnostic::SARIFDiagnostic(raw_ostream &OS, const LangOptions &LangOpts,
                                 DiagnosticOptions &DiagOpts,
                                 SarifDocumentWriter *Writer)
    : DiagnosticRenderer(LangOpts, DiagOpts),
      Root(Node::Result(), Node::Option{&LangOpts, &DiagOpts},
           /*Nesting=*/-1), // The root does not represents a diagnostic.
      Current(&Root), Writer(Writer) {
  // Don't print 'X warnings and Y errors generated'.
  DiagOpts.ShowCarets = false;
}

// helper function
namespace {
template <class NodeType, class IterateFuncType, class ApplyFuncType>
void RecursiveFor(NodeType &&Node, IterateFuncType &&IterateFunc,
                  ApplyFuncType &&ApplyFunc) {
  for (auto &&Child : IterateFunc(Node)) {
    ApplyFunc(*Child);
    RecursiveFor(*Child, IterateFunc, ApplyFunc);
  }
}
} // namespace

SARIFDiagnostic::~SARIFDiagnostic() {
  // clang-format off
  for (auto& TopLevelDiagnosticsPtr : Root.getChildrenPtrs()) { // For each top-level error/warnings.
    unsigned DiagID = TopLevelDiagnosticsPtr->getDiagID();
    SarifRule Rule = SarifRule::create() // Each top-level error/warning has a corresponding Rule.
      .setRuleId(std::to_string(DiagID))
      .setDefaultConfiguration(
        SarifReportingConfiguration::create()
          .setLevel(
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Note    ? SarifResultLevel::Note :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Remark  ? SarifResultLevel::Note :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Warning ? SarifResultLevel::Warning :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Error   ? SarifResultLevel::Error :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Fatal   ? SarifResultLevel::Error :
                                         (assert(false && "Invalid diagnostic type"), SarifResultLevel::None)
          )
          .setRank(
            TopLevelDiagnosticsPtr->getLevel() <= DiagnosticsEngine::Level::Warning ? 0  :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Error   ? 50 :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Fatal   ? 100 :
                                         (assert(false && "Invalid diagnostic type"), 0)
          )
      );
    unsigned RuleIndex = Writer->createRule(Rule); // Write into Writer.

    SarifResult Result = SarifResult::create(RuleIndex)
      .setDiagnosticMessage(TopLevelDiagnosticsPtr->getDiagnosticMessage())
      .addLocations(TopLevelDiagnosticsPtr->getLocations())
      .addRelatedLocations(TopLevelDiagnosticsPtr->getRelatedLocations());
    RecursiveFor(*TopLevelDiagnosticsPtr, [] (Node& Node) -> auto& { return Node.getChildrenPtrs(); }, [&] (Node& Node) { // For each (recursive) ChildResults.
      Result.addRelatedLocations({
        SarifChildResult::create()
          .setDiagnosticMessage(Node.getDiagnosticMessage())
          .addLocations(Node.getLocations())
          .setNesting(Node.getNesting())
      });
      Result.addRelatedLocations(Node.getRelatedLocations());
    });
    Writer->appendResult(Result); // Write into Writer
  }
  // clang-format on
}

void SARIFDiagnostic::emitDiagnosticMessage(
    FullSourceLoc Loc, PresumedLoc PLoc, DiagnosticsEngine::Level Level,
    StringRef Message, ArrayRef<clang::CharSourceRange> Ranges,
    DiagOrStoredDiag Diag) {

  if (Level >= DiagnosticsEngine::Level::Warning) {
    Current =
        &Root; // If this is a top-level error/warning, repoint Current to Root.
  } else {
    if (Message.starts_with("candidate"))
      Current =
          &Current
               ->getForkableParent(); // If this is an forked-case note, repoint
                                      // Current to the nearest forkable Node.
  }
  Current = &Current->addChildResult(
      Node::Result{Level, std::string(Message),
                   Diag}); // add child to the parent error/warning/note Node.
  Current = &Current->addLocation(
      Node::Location{Loc, PLoc, llvm::SmallVector<CharSourceRange>(Ranges)});
}

void SARIFDiagnostic::emitIncludeLocation(FullSourceLoc Loc, PresumedLoc PLoc) {
  Current = &Current->addRelatedLocation(Node::Location{Loc, PLoc, {}});
}

void SARIFDiagnostic::emitImportLocation(FullSourceLoc Loc, PresumedLoc PLoc,
                                         StringRef ModuleName) {
  Current = &Current->addRelatedLocation(Node::Location{Loc, PLoc, {}});
}

SARIFDiagnostic::Node::Node(Result Result_, Option Option_, int Nesting)
    : Result_(std::move(Result_)), Option_(std::move(Option_)),
      Nesting(Nesting) {}

SARIFDiagnostic::Node &SARIFDiagnostic::Node::getParent() {
  assert(ParentPtr && "getParent() of SARIFDiagnostic::Root!");
  return *ParentPtr;
}

SARIFDiagnostic::Node &SARIFDiagnostic::Node::getForkableParent() {
  Node *Ptr = this;
  while (Ptr->getLevel() <=
         DiagnosticsEngine::Note) // The forkable node here "is and only is"
                                  // warning/error/fatal.
    Ptr = &Ptr->getParent();
  return *Ptr;
}

llvm::SmallVector<std::unique_ptr<SARIFDiagnostic::Node>> &
SARIFDiagnostic::Node::getChildrenPtrs() {
  return ChildrenPtrs;
}

SARIFDiagnostic::Node &
SARIFDiagnostic::Node::addChildResult(Result ChildResult) {
  ChildrenPtrs.push_back(
      std::make_unique<Node>(Node::Result(std::move(ChildResult)),
                             Node::Option(std::move(Option_)), Nesting + 1));
  ChildrenPtrs.back()->ParentPtr = this; // I am the parent of this new child.
  return *ChildrenPtrs.back();
}

SARIFDiagnostic::Node &SARIFDiagnostic::Node::addLocation(Location Location) {
  Locations.push_back(std::move(Location));
  return *this;
}

SARIFDiagnostic::Node &
SARIFDiagnostic::Node::addRelatedLocation(Location Location) {
  RelatedLocations.push_back(std::move(Location));
  return *this;
}

unsigned SARIFDiagnostic::Node::getDiagID() {
  return llvm::isa<const Diagnostic *>(Result_.Diag)
             ? Result_.Diag.dyn_cast<const Diagnostic *>()->getID()
             : Result_.Diag.dyn_cast<const StoredDiagnostic *>()->getID();
}

DiagnosticsEngine::Level SARIFDiagnostic::Node::getLevel() {
  return Result_.Level;
}

std::string SARIFDiagnostic::Node::getDiagnosticMessage() {
  return Result_.Message;
}

llvm::SmallVector<CharSourceRange> SARIFDiagnostic::Node::getLocations() {
  llvm::SmallVector<CharSourceRange> CharSourceRanges;
  std::for_each(Locations.begin(), Locations.end(), [&](Location &Location) {
    CharSourceRanges.append(Location.getCharSourceRangesWithOption(Option_));
  });
  return CharSourceRanges;
}

llvm::SmallVector<CharSourceRange>
SARIFDiagnostic::Node::getRelatedLocations() {
  llvm::SmallVector<CharSourceRange> CharSourceRanges;
  std::for_each(RelatedLocations.begin(), RelatedLocations.end(),
                [&](Location &RelatedLocation) {
                  CharSourceRanges.append(
                      RelatedLocation.getCharSourceRangesWithOption(Option_));
                });
  return CharSourceRanges;
}

int SARIFDiagnostic::Node::getNesting() { return Nesting; }

llvm::SmallVector<CharSourceRange>
SARIFDiagnostic::Node::Location::getCharSourceRangesWithOption(Option Option) {
  SmallVector<CharSourceRange> Locations = {};

  if (PLoc.isInvalid()) {
    // FIXME(llvm-project/issues/57366): File-only locations
    // At least add the file name if available:
    FileID FID = Loc.getFileID();
    if (FID.isValid()) {
      if (OptionalFileEntryRef FE = Loc.getFileEntryRef()) {
        // EmitFilename(FE->getName(), Loc.getManager());
      }
    }
    return {};
  }

  FileID CaretFileID = Loc.getExpansionLoc().getFileID();

  for (const CharSourceRange Range : Ranges) {
    // Ignore invalid ranges.
    if (Range.isInvalid())
      continue;

    auto &SM = Loc.getManager();
    SourceLocation B = SM.getExpansionLoc(Range.getBegin());
    CharSourceRange ERange = SM.getExpansionRange(Range.getEnd());
    SourceLocation E = ERange.getEnd();
    bool IsTokenRange = ERange.isTokenRange();

    FileIDAndOffset BInfo = SM.getDecomposedLoc(B);
    FileIDAndOffset EInfo = SM.getDecomposedLoc(E);

    // If the start or end of the range is in another file, just discard
    // it.
    if (BInfo.first != CaretFileID || EInfo.first != CaretFileID)
      continue;

    // add in the length of the token, so that we cover multi-char
    // tokens.
    unsigned TokSize = 0;
    if (IsTokenRange)
      TokSize = Lexer::MeasureTokenLength(E, SM, *Option.LangOptsPtr);

    FullSourceLoc BF(B, SM), EF(E, SM);
    SourceLocation BeginLoc = SM.translateLineCol(
        BF.getFileID(), BF.getLineNumber(), BF.getColumnNumber());
    SourceLocation EndLoc = SM.translateLineCol(
        EF.getFileID(), EF.getLineNumber(), EF.getColumnNumber() + TokSize);

    Locations.push_back(
        CharSourceRange{SourceRange{BeginLoc, EndLoc}, /* ITR = */ false});
    // FIXME: additional ranges should use presumed location in both
    // Text and SARIF diagnostics.
  }

  auto &SM = Loc.getManager();
  auto FID = PLoc.getFileID();
  // Visual Studio 2010 or earlier expects column number to be off by one.
  unsigned int ColNo =
      (Option.LangOptsPtr->MSCompatibilityVersion &&
       !Option.LangOptsPtr->isCompatibleWithMSVC(LangOptions::MSVC2012))
          ? PLoc.getColumn() - 1
          : PLoc.getColumn();
  SourceLocation DiagLoc = SM.translateLineCol(FID, PLoc.getLine(), ColNo);

  // FIXME(llvm-project/issues/57366): Properly process #line directives.
  CharSourceRange Range = {SourceRange{DiagLoc, DiagLoc}, /* ITR = */ false};
  if (Range.isValid())
    Locations.push_back(std::move(Range));

  return Locations;
}

// llvm::StringRef SARIFDiagnostic::EmitFilename(StringRef Filename,
//                                               const SourceManager &SM) {
//   if (DiagOpts.AbsolutePath) {
//     auto File = SM.getFileManager().getOptionalFileRef(Filename);
//     if (File) {
//       // We want to print a simplified absolute path, i. e. without "dots".
//       //
//       // The hardest part here are the paths like
//       "<part1>/<link>/../<part2>".
//       // On Unix-like systems, we cannot just collapse "<link>/..", because
//       // paths are resolved sequentially, and, thereby, the path
//       // "<part1>/<part2>" may point to a different location. That is why
//       // we use FileManager::getCanonicalName(), which expands all
//       indirections
//       // with llvm::sys::fs::real_path() and caches the result.
//       //
//       // On the other hand, it would be better to preserve as much of the
//       // original path as possible, because that helps a user to recognize
//       it.
//       // real_path() expands all links, which is sometimes too much. Luckily,
//       // on Windows we can just use llvm::sys::path::remove_dots(), because,
//       // on that system, both aforementioned paths point to the same place.
// #ifdef _WIN32
//       SmallString<256> TmpFilename = File->getName();
//       llvm::sys::fs::make_absolute(TmpFilename);
//       llvm::sys::path::native(TmpFilename);
//       llvm::sys::path::remove_dots(TmpFilename, /* remove_dot_dot */ true);
//       Filename = StringRef(TmpFilename.data(), TmpFilename.size());
// #else
//       Filename = SM.getFileManager().getCanonicalName(*File);
// #endif
//     }
//   }

//   return Filename;
// }

/// Print out the file/line/column information and include trace.
///
/// This method handlen the emission of the diagnostic location information.
/// This includes extracting as much location information as is present for
/// the diagnostic and printing it, as well as any include stack or source
/// ranges necessary.
void SARIFDiagnostic::emitDiagnosticLoc(FullSourceLoc Loc, PresumedLoc PLoc,
                                        DiagnosticsEngine::Level Level,
                                        ArrayRef<CharSourceRange> Ranges) {
  assert(false && "Not implemented in SARIF mode");
}

void SARIFDiagnostic::emitBuildingModuleLocation(FullSourceLoc Loc,
                                                 PresumedLoc PLoc,
                                                 StringRef ModuleName) {
  assert(false && "Not implemented in SARIF mode");
}
} // namespace clang
