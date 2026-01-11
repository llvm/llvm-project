//===--------- SARIFDiagnostic.cpp - SARIF Diagnostic Formatting ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/SARIFDiagnostic.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/DiagnosticRenderer.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Locale.h"
#include <string>
#include <unordered_set>

namespace clang {

SARIFDiagnostic::SARIFDiagnostic(raw_ostream &OS, const LangOptions &LangOpts,
                                 DiagnosticOptions &DiagOpts,
                                 SarifDocumentWriter *Writer)
    : DiagnosticRenderer(LangOpts, DiagOpts),
      Root(Node::Result{/*Level=*/DiagnosticsEngine::Level::Error,
                        /*Message=*/"", /*Diag=*/nullptr},
           /*Nesting=*/-1), // The root does not represents a diagnostic.
      Current(&Root), LangOptsPtr(&LangOpts), Writer(Writer) {}

void SARIFDiagnostic::writeResult() {
  // clang-format off
  for (auto& TopLevelDiagnosticsPtr : Root.getChildrenPtrs()) { // For each top-level error/warnings.
    unsigned DiagID = TopLevelDiagnosticsPtr->getDiagID();
    SarifRule Rule = SarifRule::create() // Each top-level error/warning has a corresponding Rule.
      .setRuleId(std::to_string(DiagID))
      .setDefaultConfiguration(
        SarifReportingConfiguration::create()
          .setLevel(
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Note    ? SarifResultLevel::Note    :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Remark  ? SarifResultLevel::Note    :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Warning ? SarifResultLevel::Warning :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Error   ? SarifResultLevel::Error   :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Fatal   ? SarifResultLevel::Error   :
                                         (assert(false && "Invalid diagnostic type"), SarifResultLevel::None)
          )
          .setRank(
            TopLevelDiagnosticsPtr->getLevel() <= DiagnosticsEngine::Level::Warning ? 0   :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Error   ? 50  :
            TopLevelDiagnosticsPtr->getLevel() == DiagnosticsEngine::Level::Fatal   ? 100 :
                                         (assert(false && "Invalid diagnostic type"), 0)
          )
      );
    unsigned RuleIndex = Writer->createRule(Rule); // Write into Writer.

    SarifResult Result = SarifResult::create(RuleIndex)
      .setDiagnosticMessage(TopLevelDiagnosticsPtr->getDiagnosticMessage())
      .addLocations(TopLevelDiagnosticsPtr->getLocations(*LangOptsPtr))
      .addRelatedLocations(TopLevelDiagnosticsPtr->getRelatedLocations(*LangOptsPtr));
    TopLevelDiagnosticsPtr->recursiveForEach([&] (Node& Node) { // For each (recursive) ChildResults.
      Result.addRelatedLocations({
        SarifChildResult::create()
          .setDiagnosticMessage(Node.getDiagnosticMessage())
          .addLocations(Node.getLocations(*LangOptsPtr))
          .setNesting(Node.getNesting())
      });
      Result.addRelatedLocations(Node.getRelatedLocations(*LangOptsPtr));
    });
    Writer->appendResult(Result); // Write into Writer.
  }
  // clang-format on
  Root.getChildrenPtrs().clear(); // Reset the result cache
}

void SARIFDiagnostic::setLangOptions(const LangOptions &LangOpts) {
  LangOptsPtr = &LangOpts;
}

void SARIFDiagnostic::emitInvocation(CompilerInstance &Compiler,
                                     bool Successful, StringRef Message) {
  Writer->appendInvocation(
      /*CommandLine=*/Compiler.getInvocation().getCC1CommandLine(),
      /*ExecutionSuccessful=*/Successful,
      /*ToolExecutionNotification=*/Message);
}

void SARIFDiagnostic::emitDiagnosticMessage(
    FullSourceLoc Loc, PresumedLoc PLoc, DiagnosticsEngine::Level Level,
    StringRef Message, ArrayRef<clang::CharSourceRange> Ranges,
    DiagOrStoredDiag Diag) {
  if (Level >= DiagnosticsEngine::Level::Warning) {
    Current =
        &Root; // If this is a top-level error/warning, repoint Current to Root.
  } else {
    auto ID = llvm::isa<const Diagnostic*>(Diag) ? Diag.dyn_cast<const Diagnostic*>()->getID() : 
                                       Diag.dyn_cast<const StoredDiagnostic*>()->getID();
    if (ForkableDiagIDs.find(ID) != ForkableDiagIDs.end() or 
        Message.starts_with("candidate"))
      Current =
          &Current
               ->getForkableParent(); // If this is an forked-case note, repoint
                                      // Current to the nearest forkable Node.
  }
  Current = &Current->addChildResult(
      Node::Result{Level, std::string(Message),
                   Diag}); // Add child to the parent error/warning/note Node.
  Current = &Current->addLocation(
      Node::Location{Loc, PLoc, llvm::SmallVector<CharSourceRange>(Ranges)});
}

void SARIFDiagnostic::emitIncludeLocation(FullSourceLoc Loc, PresumedLoc PLoc) {
  Current =
      &Current->addRelatedLocation(Node::Location{Loc, PLoc, /*Ranges=*/{}});
}

void SARIFDiagnostic::emitImportLocation(FullSourceLoc Loc, PresumedLoc PLoc,
                                         StringRef ModuleName) {
  Current = &Current->addRelatedLocation(Node::Location{Loc, PLoc, {}});
}

SARIFDiagnostic::Node::Node(Result Result_, int Nesting)
    : Result_(std::move(Result_)), Nesting(Nesting) {}

SARIFDiagnostic::Node &SARIFDiagnostic::Node::getParent() {
  assert(ParentPtr && "getParent() of SARIFDiagnostic::Root!");
  return *ParentPtr;
}

SARIFDiagnostic::Node &SARIFDiagnostic::Node::getForkableParent() {
  Node *Ptr = this;
  // The forkable node here "is and only is" warning/error/fatal.
  while (Ptr->getLevel() <= DiagnosticsEngine::Note)
    Ptr = &Ptr->getParent();
  return *Ptr;
}

llvm::SmallVector<std::unique_ptr<SARIFDiagnostic::Node>> &
SARIFDiagnostic::Node::getChildrenPtrs() {
  return ChildrenPtrs;
}

SARIFDiagnostic::Node &
SARIFDiagnostic::Node::addChildResult(Result ChildResult) {
  ChildrenPtrs.push_back(std::make_unique<Node>(
      Node::Result(std::move(ChildResult)), Nesting + 1));
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

template <class Func>
void SARIFDiagnostic::Node::recursiveForEach(Func &&Function) {
  for (auto &&ChildPtr : getChildrenPtrs()) {
    Function(*ChildPtr);
    ChildPtr->recursiveForEach(std::forward<Func &&>(Function));
  }
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

llvm::SmallVector<CharSourceRange>
SARIFDiagnostic::Node::getLocations(const LangOptions &LangOpts) {
  llvm::SmallVector<CharSourceRange> CharSourceRanges;
  llvm::for_each(Locations, [&](Location &Location) {
    CharSourceRanges.append(Location.getCharSourceRangesWithOption(LangOpts));
  });
  return CharSourceRanges;
}

llvm::SmallVector<CharSourceRange>
SARIFDiagnostic::Node::getRelatedLocations(const LangOptions &LangOpts) {
  llvm::SmallVector<CharSourceRange> CharSourceRanges;
  llvm::for_each(RelatedLocations, [&](Location &RelatedLocation) {
    CharSourceRanges.append(
        RelatedLocation.getCharSourceRangesWithOption(LangOpts));
  });
  return CharSourceRanges;
}

int SARIFDiagnostic::Node::getNesting() { return Nesting; }

llvm::SmallVector<CharSourceRange>
SARIFDiagnostic::Node::Location::getCharSourceRangesWithOption(
    const LangOptions &LangOpts) {
  SmallVector<CharSourceRange> Locations = {};

  if (PLoc.isInvalid()) {
    // FIXME(llvm-project/issues/57366): File-only locations
    // At least add the file name if available.
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

    // Add in the length of the token, so that we cover multi-char
    // tokens.
    unsigned TokSize = 0;
    if (IsTokenRange)
      TokSize = Lexer::MeasureTokenLength(E, SM, LangOpts);

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
  unsigned int ColNo = (LangOpts.MSCompatibilityVersion &&
                        !LangOpts.isCompatibleWithMSVC(LangOptions::MSVC2012))
                           ? PLoc.getColumn() - 1
                           : PLoc.getColumn();
  SourceLocation DiagLoc = SM.translateLineCol(FID, PLoc.getLine(), ColNo);

  // FIXME(llvm-project/issues/57366): Properly process #line directives.
  CharSourceRange Range = {SourceRange{DiagLoc, DiagLoc}, /* ITR = */ false};
  if (Range.isValid())
    Locations.push_back(std::move(Range));

  return Locations;
}

std::unordered_set<unsigned> SARIFDiagnostic::ForkableDiagIDs = {
  // Overload
  diag::note_ovl_too_many_candidates,
  diag::note_ovl_candidate,
  diag::note_ovl_candidate_explicit,
  diag::note_ovl_candidate_inherited_constructor,
  diag::note_ovl_candidate_inherited_constructor_slice,
  diag::note_ovl_candidate_illegal_constructor,
  diag::note_ovl_candidate_illegal_constructor_adrspace_mismatch,
  diag::note_ovl_candidate_bad_deduction,
  diag::note_ovl_candidate_incomplete_deduction,
  diag::note_ovl_candidate_incomplete_deduction_pack,
  diag::note_ovl_candidate_inconsistent_deduction,
  diag::note_ovl_candidate_inconsistent_deduction_types,
  diag::note_ovl_candidate_explicit_arg_mismatch_named,
  diag::note_ovl_candidate_unsatisfied_constraints,
  diag::note_ovl_candidate_explicit_arg_mismatch_unnamed,
  diag::note_ovl_candidate_instantiation_depth,
  diag::note_ovl_candidate_underqualified,
  diag::note_ovl_candidate_substitution_failure,
  diag::note_ovl_candidate_disabled_by_enable_if,
  diag::note_ovl_candidate_disabled_by_requirement,
  diag::note_ovl_candidate_has_pass_object_size_params,
  diag::note_ovl_candidate_disabled_by_function_cond_attr,
  diag::note_ovl_candidate_deduced_mismatch,
  diag::note_ovl_candidate_non_deduced_mismatch,
  diag::note_ovl_candidate_non_deduced_mismatch_qualified,
  diag::note_ovl_candidate_arity,
  diag::note_ovl_candidate_arity_one,
  diag::note_ovl_candidate_deleted,
  diag::note_ovl_candidate_bad_conv_incomplete,
  diag::note_ovl_candidate_bad_list_argument,
  diag::note_ovl_candidate_bad_overload,
  diag::note_ovl_candidate_bad_conv,
  diag::note_ovl_candidate_bad_arc_conv,
  diag::note_ovl_candidate_bad_value_category,
  diag::note_ovl_candidate_bad_addrspace,
  diag::note_ovl_candidate_bad_addrspace_this,
  diag::note_ovl_candidate_bad_gc,
  diag::note_ovl_candidate_bad_ownership,
  diag::note_ovl_candidate_bad_ptrauth,
  diag::note_ovl_candidate_bad_cvr_this,
  diag::note_ovl_candidate_bad_cvr,
  diag::note_ovl_candidate_bad_base_to_derived_conv,
  diag::note_ovl_candidate_bad_target,
  diag::note_ovl_candidate_constraints_not_satisfied,
  diag::note_ovl_surrogate_constraints_not_satisfied,
  diag::note_ovl_builtin_candidate,
  diag::note_ovl_ambiguous_oper_binary_reversed_self,
  diag::note_ovl_ambiguous_eqeq_reversed_self_non_const,
  diag::note_ovl_ambiguous_oper_binary_selected_candidate,
  diag::note_ovl_ambiguous_oper_binary_reversed_candidate,
  diag::note_ovl_surrogate_cand,
  
  // Multi-declaration/definition
  diag::note_previous_declaration,
  diag::note_previous_definition,
  diag::note_declared_at,

  // Base-derived reason list.
  diag::note_unsatisfied_trait_reason,
  diag::note_overridden_virtual_function,

  // See clang/lib/Sema/SemaLookup.cpp -> DiagnoseAmbiguousLookup()
  diag::note_ambiguous_candidate,
  diag::note_ambiguous_member_found,
  diag::note_ambiguous_member_type_found,

  // See clang/lib/Sema/Sema.cpp -> noteOverloads()
  diag::note_possible_target_of_call,
  
  // See clang/lib/Sema/SemaStmt.cpp -> ActOnFinishSwitchStmt()
  diag::note_duplicate_case_prev,
};

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
