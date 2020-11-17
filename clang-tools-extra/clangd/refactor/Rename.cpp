//===--- Rename.cpp - Symbol-rename refactorings -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/Rename.h"
#include "AST.h"
#include "FindTarget.h"
#include "ParsedAST.h"
#include "Selection.h"
#include "SourceCode.h"
#include "index/SymbolCollector.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Refactoring/Rename/USRFindingAction.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>

namespace clang {
namespace clangd {
namespace {

llvm::Optional<std::string> filePath(const SymbolLocation &Loc,
                                     llvm::StringRef HintFilePath) {
  if (!Loc)
    return None;
  auto Path = URI::resolve(Loc.FileURI, HintFilePath);
  if (!Path) {
    elog("Could not resolve URI {0}: {1}", Loc.FileURI, Path.takeError());
    return None;
  }

  return *Path;
}

// Returns true if the given location is expanded from any macro body.
bool isInMacroBody(const SourceManager &SM, SourceLocation Loc) {
  while (Loc.isMacroID()) {
    if (SM.isMacroBodyExpansion(Loc))
      return true;
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }

  return false;
}

// Query the index to find some other files where the Decl is referenced.
llvm::Optional<std::string> getOtherRefFile(const Decl &D, StringRef MainFile,
                                            const SymbolIndex &Index) {
  RefsRequest Req;
  // We limit the number of results, this is a correctness/performance
  // tradeoff. We expect the number of symbol references in the current file
  // is smaller than the limit.
  Req.Limit = 100;
  Req.IDs.insert(getSymbolID(&D));
  llvm::Optional<std::string> OtherFile;
  Index.refs(Req, [&](const Ref &R) {
    if (OtherFile)
      return;
    if (auto RefFilePath = filePath(R.Location, /*HintFilePath=*/MainFile)) {
      if (*RefFilePath != MainFile)
        OtherFile = *RefFilePath;
    }
  });
  return OtherFile;
}

llvm::DenseSet<const NamedDecl *> locateDeclAt(ParsedAST &AST,
                                               SourceLocation TokenStartLoc) {
  unsigned Offset =
      AST.getSourceManager().getDecomposedSpellingLoc(TokenStartLoc).second;

  SelectionTree Selection = SelectionTree::createRight(
      AST.getASTContext(), AST.getTokens(), Offset, Offset);
  const SelectionTree::Node *SelectedNode = Selection.commonAncestor();
  if (!SelectedNode)
    return {};

  llvm::DenseSet<const NamedDecl *> Result;
  for (const NamedDecl *D :
       targetDecl(SelectedNode->ASTNode,
                  DeclRelation::Alias | DeclRelation::TemplatePattern)) {
    // Get to CXXRecordDecl from constructor or destructor.
    D = tooling::getCanonicalSymbolDeclaration(D);
    Result.insert(D);
  }
  return Result;
}

// By default, we exclude C++ standard symbols and protobuf symbols as rename
// these symbols would change system/generated files which are unlikely to be
// modified.
bool isExcluded(const NamedDecl &RenameDecl) {
  if (isProtoFile(RenameDecl.getLocation(),
                  RenameDecl.getASTContext().getSourceManager()))
    return true;
  static const auto *StdSymbols = new llvm::DenseSet<llvm::StringRef>({
#define SYMBOL(Name, NameSpace, Header) {#NameSpace #Name},
#include "StdSymbolMap.inc"
#undef SYMBOL
  });
  return StdSymbols->count(printQualifiedName(RenameDecl));
}

enum class ReasonToReject {
  NoSymbolFound,
  NoIndexProvided,
  NonIndexable,
  UsedOutsideFile, // for within-file rename only.
  UnsupportedSymbol,
  AmbiguousSymbol,

  // name validation.
  RenameToKeywords,
  SameName,
};

llvm::Optional<ReasonToReject> renameable(const NamedDecl &RenameDecl,
                                          StringRef MainFilePath,
                                          const SymbolIndex *Index,
                                          bool CrossFile) {
  trace::Span Tracer("Renameable");
  // Filter out symbols that are unsupported in both rename modes.
  if (llvm::isa<NamespaceDecl>(&RenameDecl))
    return ReasonToReject::UnsupportedSymbol;
  if (const auto *FD = llvm::dyn_cast<FunctionDecl>(&RenameDecl)) {
    if (FD->isOverloadedOperator())
      return ReasonToReject::UnsupportedSymbol;
  }
  // function-local symbols is safe to rename.
  if (RenameDecl.getParentFunctionOrMethod())
    return None;

  if (isExcluded(RenameDecl))
    return ReasonToReject::UnsupportedSymbol;

  // Check whether the symbol being rename is indexable.
  auto &ASTCtx = RenameDecl.getASTContext();
  bool MainFileIsHeader = isHeaderFile(MainFilePath, ASTCtx.getLangOpts());
  bool DeclaredInMainFile =
      isInsideMainFile(RenameDecl.getBeginLoc(), ASTCtx.getSourceManager());
  bool IsMainFileOnly = true;
  if (MainFileIsHeader)
    // main file is a header, the symbol can't be main file only.
    IsMainFileOnly = false;
  else if (!DeclaredInMainFile)
    IsMainFileOnly = false;
  // If the symbol is not indexable, we disallow rename.
  if (!SymbolCollector::shouldCollectSymbol(
          RenameDecl, RenameDecl.getASTContext(), SymbolCollector::Options(),
          IsMainFileOnly))
    return ReasonToReject::NonIndexable;

  if (!CrossFile) {
    if (!DeclaredInMainFile)
      // We are sure the symbol is used externally, bail out early.
      return ReasonToReject::UsedOutsideFile;

    // If the symbol is declared in the main file (which is not a header), we
    // rename it.
    if (!MainFileIsHeader)
      return None;

    if (!Index)
      return ReasonToReject::NoIndexProvided;

    auto OtherFile = getOtherRefFile(RenameDecl, MainFilePath, *Index);
    // If the symbol is indexable and has no refs from other files in the index,
    // we rename it.
    if (!OtherFile)
      return None;
    // If the symbol is indexable and has refs from other files in the index,
    // we disallow rename.
    return ReasonToReject::UsedOutsideFile;
  }

  assert(CrossFile);

  // FIXME: Renaming virtual methods requires to rename all overridens in
  // subclasses, our index doesn't have this information.
  // Note: Within-file rename does support this through the AST.
  if (const auto *S = llvm::dyn_cast<CXXMethodDecl>(&RenameDecl)) {
    if (S->isVirtual())
      return ReasonToReject::UnsupportedSymbol;
  }
  return None;
}

llvm::Error makeError(ReasonToReject Reason) {
  auto Message = [](ReasonToReject Reason) {
    switch (Reason) {
    case ReasonToReject::NoSymbolFound:
      return "there is no symbol at the given location";
    case ReasonToReject::NoIndexProvided:
      return "no index provided";
    case ReasonToReject::UsedOutsideFile:
      return "the symbol is used outside main file";
    case ReasonToReject::NonIndexable:
      return "symbol may be used in other files (not eligible for indexing)";
    case ReasonToReject::UnsupportedSymbol:
      return "symbol is not a supported kind (e.g. namespace, macro)";
    case ReasonToReject::AmbiguousSymbol:
      return "there are multiple symbols at the given location";
    case ReasonToReject::RenameToKeywords:
      return "the chosen name is a keyword";
    case ReasonToReject::SameName:
      return "new name is the same as the old name";
    }
    llvm_unreachable("unhandled reason kind");
  };
  return error("Cannot rename symbol: {0}", Message(Reason));
}

// Return all rename occurrences in the main file.
std::vector<SourceLocation> findOccurrencesWithinFile(ParsedAST &AST,
                                                      const NamedDecl &ND) {
  trace::Span Tracer("FindOccurrencesWithinFile");
  // If the cursor is at the underlying CXXRecordDecl of the
  // ClassTemplateDecl, ND will be the CXXRecordDecl. In this case, we need to
  // get the primary template manually.
  // getUSRsForDeclaration will find other related symbols, e.g. virtual and its
  // overriddens, primary template and all explicit specializations.
  // FIXME: Get rid of the remaining tooling APIs.
  const auto *RenameDecl =
      ND.getDescribedTemplate() ? ND.getDescribedTemplate() : &ND;
  std::vector<std::string> RenameUSRs =
      tooling::getUSRsForDeclaration(RenameDecl, AST.getASTContext());
  llvm::DenseSet<SymbolID> TargetIDs;
  for (auto &USR : RenameUSRs)
    TargetIDs.insert(SymbolID(USR));

  std::vector<SourceLocation> Results;
  for (Decl *TopLevelDecl : AST.getLocalTopLevelDecls()) {
    findExplicitReferences(TopLevelDecl, [&](ReferenceLoc Ref) {
      if (Ref.Targets.empty())
        return;
      for (const auto *Target : Ref.Targets) {
        auto ID = getSymbolID(Target);
        if (!ID || TargetIDs.find(ID) == TargetIDs.end())
          return;
      }
      Results.push_back(Ref.NameLoc);
    });
  }

  return Results;
}

// Lookup the declarations (if any) with the given Name in the context of
// RenameDecl.
const NamedDecl *lookupSiblingWithName(const ASTContext &Ctx,
                                       const NamedDecl &RenamedDecl,
                                       llvm::StringRef Name) {
  const auto &II = Ctx.Idents.get(Name);
  DeclarationName LookupName(&II);
  DeclContextLookupResult LookupResult;
  const auto *DC = RenamedDecl.getDeclContext();
  while (DC && DC->isTransparentContext())
    DC = DC->getParent();
  switch (DC->getDeclKind()) {
  // The enclosing DeclContext may not be the enclosing scope, it might have
  // false positives and negatives, so we only choose "confident" DeclContexts
  // that don't have any subscopes that are neither DeclContexts nor
  // transparent.
  //
  // Notably, FunctionDecl is excluded -- because local variables are not scoped
  // to the function, but rather to the CompoundStmt that is its body. Lookup
  // will not find function-local variables.
  case Decl::TranslationUnit:
  case Decl::Namespace:
  case Decl::Record:
  case Decl::Enum:
  case Decl::CXXRecord:
    LookupResult = DC->lookup(LookupName);
    break;
  default:
    break;
  }
  // Lookup may contain the RenameDecl itself, exclude it.
  for (const auto *D : LookupResult)
    if (D->getCanonicalDecl() != RenamedDecl.getCanonicalDecl())
      return D;
  return nullptr;
}

struct InvalidName {
  enum Kind {
    Keywords,
    Conflict,
  };
  Kind K;
  std::string Details;
};

llvm::Error makeError(InvalidName Reason) {
  auto Message = [](InvalidName Reason) {
    switch (Reason.K) {
    case InvalidName::Keywords:
      return llvm::formatv("the chosen name \"{0}\" is a keyword",
                           Reason.Details);
    case InvalidName::Conflict:
      return llvm::formatv("conflict with the symbol in {0}", Reason.Details);
    }
    llvm_unreachable("unhandled InvalidName kind");
  };
  return error("invalid name: {0}", Message(Reason));
}

// Check if we can rename the given RenameDecl into NewName.
// Return details if the rename would produce a conflict.
llvm::Optional<InvalidName> checkName(const NamedDecl &RenameDecl,
                                      llvm::StringRef NewName) {
  auto &ASTCtx = RenameDecl.getASTContext();
  if (isKeyword(NewName, ASTCtx.getLangOpts()))
    return InvalidName{InvalidName::Keywords, NewName.str()};
  // Name conflict detection.
  // Function conflicts are subtle (overloading), so ignore them.
  if (RenameDecl.getKind() != Decl::Function) {
    if (auto *Conflict = lookupSiblingWithName(ASTCtx, RenameDecl, NewName))
      return InvalidName{
          InvalidName::Conflict,
          Conflict->getLocation().printToString(ASTCtx.getSourceManager())};
  }
  return llvm::None;
}

// AST-based rename, it renames all occurrences in the main file.
llvm::Expected<tooling::Replacements>
renameWithinFile(ParsedAST &AST, const NamedDecl &RenameDecl,
                 llvm::StringRef NewName) {
  trace::Span Tracer("RenameWithinFile");
  const SourceManager &SM = AST.getSourceManager();

  tooling::Replacements FilteredChanges;
  for (SourceLocation Loc : findOccurrencesWithinFile(AST, RenameDecl)) {
    SourceLocation RenameLoc = Loc;
    // We don't rename in any macro bodies, but we allow rename the symbol
    // spelled in a top-level macro argument in the main file.
    if (RenameLoc.isMacroID()) {
      if (isInMacroBody(SM, RenameLoc))
        continue;
      RenameLoc = SM.getSpellingLoc(Loc);
    }
    // Filter out locations not from main file.
    // We traverse only main file decls, but locations could come from an
    // non-preamble #include file e.g.
    //   void test() {
    //     int f^oo;
    //     #include "use_foo.inc"
    //   }
    if (!isInsideMainFile(RenameLoc, SM))
      continue;
    if (auto Err = FilteredChanges.add(tooling::Replacement(
            SM, CharSourceRange::getTokenRange(RenameLoc), NewName)))
      return std::move(Err);
  }
  return FilteredChanges;
}

Range toRange(const SymbolLocation &L) {
  Range R;
  R.start.line = L.Start.line();
  R.start.character = L.Start.column();
  R.end.line = L.End.line();
  R.end.character = L.End.column();
  return R;
}

// Return all rename occurrences (using the index) outside of the main file,
// grouped by the absolute file path.
llvm::Expected<llvm::StringMap<std::vector<Range>>>
findOccurrencesOutsideFile(const NamedDecl &RenameDecl,
                           llvm::StringRef MainFile, const SymbolIndex &Index,
                           size_t MaxLimitFiles) {
  trace::Span Tracer("FindOccurrencesOutsideFile");
  RefsRequest RQuest;
  RQuest.IDs.insert(getSymbolID(&RenameDecl));

  // Absolute file path => rename occurrences in that file.
  llvm::StringMap<std::vector<Range>> AffectedFiles;
  bool HasMore = Index.refs(RQuest, [&](const Ref &R) {
    if (AffectedFiles.size() >= MaxLimitFiles)
      return;
    if ((R.Kind & RefKind::Spelled) == RefKind::Unknown)
      return;
    if (auto RefFilePath = filePath(R.Location, /*HintFilePath=*/MainFile)) {
      if (*RefFilePath != MainFile)
        AffectedFiles[*RefFilePath].push_back(toRange(R.Location));
    }
  });

  if (AffectedFiles.size() >= MaxLimitFiles)
    return error("The number of affected files exceeds the max limit {0}",
                 MaxLimitFiles);
  if (HasMore)
    return error("The symbol {0} has too many occurrences",
                 RenameDecl.getQualifiedNameAsString());
  // Sort and deduplicate the results, in case that index returns duplications.
  for (auto &FileAndOccurrences : AffectedFiles) {
    auto &Ranges = FileAndOccurrences.getValue();
    llvm::sort(Ranges);
    Ranges.erase(std::unique(Ranges.begin(), Ranges.end()), Ranges.end());

    SPAN_ATTACH(Tracer, FileAndOccurrences.first(),
                static_cast<int64_t>(Ranges.size()));
  }
  return AffectedFiles;
}

// Index-based rename, it renames all occurrences outside of the main file.
//
// The cross-file rename is purely based on the index, as we don't want to
// build all ASTs for affected files, which may cause a performance hit.
// We choose to trade off some correctness for performance and scalability.
//
// Clangd builds a dynamic index for all opened files on top of the static
// index of the whole codebase. Dynamic index is up-to-date (respects dirty
// buffers) as long as clangd finishes processing opened files, while static
// index (background index) is relatively stale. We choose the dirty buffers
// as the file content we rename on, and fallback to file content on disk if
// there is no dirty buffer.
llvm::Expected<FileEdits> renameOutsideFile(
    const NamedDecl &RenameDecl, llvm::StringRef MainFilePath,
    llvm::StringRef NewName, const SymbolIndex &Index, size_t MaxLimitFiles,
    llvm::function_ref<llvm::Expected<std::string>(PathRef)> GetFileContent) {
  trace::Span Tracer("RenameOutsideFile");
  auto AffectedFiles = findOccurrencesOutsideFile(RenameDecl, MainFilePath,
                                                  Index, MaxLimitFiles);
  if (!AffectedFiles)
    return AffectedFiles.takeError();
  FileEdits Results;
  for (auto &FileAndOccurrences : *AffectedFiles) {
    llvm::StringRef FilePath = FileAndOccurrences.first();

    auto AffectedFileCode = GetFileContent(FilePath);
    if (!AffectedFileCode) {
      elog("Fail to read file content: {0}", AffectedFileCode.takeError());
      continue;
    }
    auto RenameRanges =
        adjustRenameRanges(*AffectedFileCode, RenameDecl.getNameAsString(),
                           std::move(FileAndOccurrences.second),
                           RenameDecl.getASTContext().getLangOpts());
    if (!RenameRanges) {
      // Our heuristics fails to adjust rename ranges to the current state of
      // the file, it is most likely the index is stale, so we give up the
      // entire rename.
      return error("Index results don't match the content of file {0} "
                   "(the index may be stale)",
                   FilePath);
    }
    auto RenameEdit =
        buildRenameEdit(FilePath, *AffectedFileCode, *RenameRanges, NewName);
    if (!RenameEdit)
      return error("failed to rename in file {0}: {1}", FilePath,
                   RenameEdit.takeError());
    if (!RenameEdit->Replacements.empty())
      Results.insert({FilePath, std::move(*RenameEdit)});
  }
  return Results;
}

// A simple edit is either changing line or column, but not both.
bool impliesSimpleEdit(const Position &LHS, const Position &RHS) {
  return LHS.line == RHS.line || LHS.character == RHS.character;
}

// Performs a DFS to enumerate all possible near-miss matches.
// It finds the locations where the indexed occurrences are now spelled in
// Lexed occurrences, a near miss is defined as:
//   - a near miss maps all of the **name** occurrences from the index onto a
//     *subset* of lexed occurrences (we allow a single name refers to more
//     than one symbol)
//   - all indexed occurrences must be mapped, and Result must be distinct and
//     preserve order (only support detecting simple edits to ensure a
//     robust mapping)
//   - each indexed -> lexed occurrences mapping correspondence may change the
//     *line* or *column*, but not both (increases chance of a robust mapping)
void findNearMiss(
    std::vector<size_t> &PartialMatch, ArrayRef<Range> IndexedRest,
    ArrayRef<Range> LexedRest, int LexedIndex, int &Fuel,
    llvm::function_ref<void(const std::vector<size_t> &)> MatchedCB) {
  if (--Fuel < 0)
    return;
  if (IndexedRest.size() > LexedRest.size())
    return;
  if (IndexedRest.empty()) {
    MatchedCB(PartialMatch);
    return;
  }
  if (impliesSimpleEdit(IndexedRest.front().start, LexedRest.front().start)) {
    PartialMatch.push_back(LexedIndex);
    findNearMiss(PartialMatch, IndexedRest.drop_front(), LexedRest.drop_front(),
                 LexedIndex + 1, Fuel, MatchedCB);
    PartialMatch.pop_back();
  }
  findNearMiss(PartialMatch, IndexedRest, LexedRest.drop_front(),
               LexedIndex + 1, Fuel, MatchedCB);
}

} // namespace

llvm::Expected<RenameResult> rename(const RenameInputs &RInputs) {
  trace::Span Tracer("Rename flow");
  const auto &Opts = RInputs.Opts;
  ParsedAST &AST = RInputs.AST;
  const SourceManager &SM = AST.getSourceManager();
  llvm::StringRef MainFileCode = SM.getBufferData(SM.getMainFileID());
  auto GetFileContent = [&RInputs,
                         &SM](PathRef AbsPath) -> llvm::Expected<std::string> {
    llvm::Optional<std::string> DirtyBuffer;
    if (RInputs.GetDirtyBuffer &&
        (DirtyBuffer = RInputs.GetDirtyBuffer(AbsPath)))
      return std::move(*DirtyBuffer);

    auto Content =
        SM.getFileManager().getVirtualFileSystem().getBufferForFile(AbsPath);
    if (!Content)
      return error("Fail to open file {0}: {1}", AbsPath,
                   Content.getError().message());
    if (!*Content)
      return error("Got no buffer for file {0}", AbsPath);

    return (*Content)->getBuffer().str();
  };
  // Try to find the tokens adjacent to the cursor position.
  auto Loc = sourceLocationInMainFile(SM, RInputs.Pos);
  if (!Loc)
    return Loc.takeError();
  const syntax::Token *IdentifierToken =
      spelledIdentifierTouching(*Loc, AST.getTokens());

  // Renames should only triggered on identifiers.
  if (!IdentifierToken)
    return makeError(ReasonToReject::NoSymbolFound);
  Range CurrentIdentifier = halfOpenToRange(
      SM, CharSourceRange::getCharRange(IdentifierToken->location(),
                                        IdentifierToken->endLocation()));
  // FIXME: Renaming macros is not supported yet, the macro-handling code should
  // be moved to rename tooling library.
  if (locateMacroAt(*IdentifierToken, AST.getPreprocessor()))
    return makeError(ReasonToReject::UnsupportedSymbol);

  auto DeclsUnderCursor = locateDeclAt(AST, IdentifierToken->location());
  if (DeclsUnderCursor.empty())
    return makeError(ReasonToReject::NoSymbolFound);
  if (DeclsUnderCursor.size() > 1)
    return makeError(ReasonToReject::AmbiguousSymbol);
  const auto &RenameDecl =
      llvm::cast<NamedDecl>(*(*DeclsUnderCursor.begin())->getCanonicalDecl());
  if (RenameDecl.getName() == RInputs.NewName)
    return makeError(ReasonToReject::SameName);
  auto Invalid = checkName(RenameDecl, RInputs.NewName);
  if (Invalid)
    return makeError(*Invalid);

  auto Reject = renameable(RenameDecl, RInputs.MainFilePath, RInputs.Index,
                           Opts.AllowCrossFile);
  if (Reject)
    return makeError(*Reject);

  // We have two implementations of the rename:
  //   - AST-based rename: used for renaming local symbols, e.g. variables
  //     defined in a function body;
  //   - index-based rename: used for renaming non-local symbols, and not
  //     feasible for local symbols (as by design our index don't index these
  //     symbols by design;
  // To make cross-file rename work for local symbol, we use a hybrid solution:
  //   - run AST-based rename on the main file;
  //   - run index-based rename on other affected files;
  auto MainFileRenameEdit = renameWithinFile(AST, RenameDecl, RInputs.NewName);
  if (!MainFileRenameEdit)
    return MainFileRenameEdit.takeError();
  RenameResult Result;
  Result.Target = CurrentIdentifier;
  Edit MainFileEdits = Edit(MainFileCode, std::move(*MainFileRenameEdit));
  llvm::for_each(MainFileEdits.asTextEdits(), [&Result](const TextEdit &TE) {
    Result.LocalChanges.push_back(TE.range);
  });

  // return the main file edit if this is a within-file rename or the symbol
  // being renamed is function local.
  if (!Opts.AllowCrossFile || RenameDecl.getParentFunctionOrMethod()) {
    Result.GlobalChanges = FileEdits(
        {std::make_pair(RInputs.MainFilePath, std::move(MainFileEdits))});
    return Result;
  }

  // If the index is nullptr, we don't know the completeness of the result, so
  // we don't populate the field GlobalChanges.
  if (!RInputs.Index) {
    assert(Result.GlobalChanges.empty() && Opts.AllowCrossFile);
    return Result;
  }

  auto OtherFilesEdits = renameOutsideFile(
      RenameDecl, RInputs.MainFilePath, RInputs.NewName, *RInputs.Index,
      Opts.LimitFiles == 0 ? std::numeric_limits<size_t>::max()
                           : Opts.LimitFiles,
      GetFileContent);
  if (!OtherFilesEdits)
    return OtherFilesEdits.takeError();
  Result.GlobalChanges = *OtherFilesEdits;
  // Attach the rename edits for the main file.
  Result.GlobalChanges.try_emplace(RInputs.MainFilePath,
                                   std::move(MainFileEdits));
  return Result;
}

llvm::Expected<Edit> buildRenameEdit(llvm::StringRef AbsFilePath,
                                     llvm::StringRef InitialCode,
                                     std::vector<Range> Occurrences,
                                     llvm::StringRef NewName) {
  trace::Span Tracer("BuildRenameEdit");
  SPAN_ATTACH(Tracer, "file_path", AbsFilePath);
  SPAN_ATTACH(Tracer, "rename_occurrences",
              static_cast<int64_t>(Occurrences.size()));

  assert(std::is_sorted(Occurrences.begin(), Occurrences.end()));
  assert(std::unique(Occurrences.begin(), Occurrences.end()) ==
             Occurrences.end() &&
         "Occurrences must be unique");

  // These two always correspond to the same position.
  Position LastPos{0, 0};
  size_t LastOffset = 0;

  auto Offset = [&](const Position &P) -> llvm::Expected<size_t> {
    assert(LastPos <= P && "malformed input");
    Position Shifted = {
        P.line - LastPos.line,
        P.line > LastPos.line ? P.character : P.character - LastPos.character};
    auto ShiftedOffset =
        positionToOffset(InitialCode.substr(LastOffset), Shifted);
    if (!ShiftedOffset)
      return error("fail to convert the position {0} to offset ({1})", P,
                   ShiftedOffset.takeError());
    LastPos = P;
    LastOffset += *ShiftedOffset;
    return LastOffset;
  };

  std::vector<std::pair</*start*/ size_t, /*end*/ size_t>> OccurrencesOffsets;
  for (const auto &R : Occurrences) {
    auto StartOffset = Offset(R.start);
    if (!StartOffset)
      return StartOffset.takeError();
    auto EndOffset = Offset(R.end);
    if (!EndOffset)
      return EndOffset.takeError();
    OccurrencesOffsets.push_back({*StartOffset, *EndOffset});
  }

  tooling::Replacements RenameEdit;
  for (const auto &R : OccurrencesOffsets) {
    auto ByteLength = R.second - R.first;
    if (auto Err = RenameEdit.add(
            tooling::Replacement(AbsFilePath, R.first, ByteLength, NewName)))
      return std::move(Err);
  }
  return Edit(InitialCode, std::move(RenameEdit));
}

// Details:
//  - lex the draft code to get all rename candidates, this yields a superset
//    of candidates.
//  - apply range patching heuristics to generate "authoritative" occurrences,
//    cases we consider:
//      (a) index returns a subset of candidates, we use the indexed results.
//        - fully equal, we are sure the index is up-to-date
//        - proper subset, index is correct in most cases? there may be false
//          positives (e.g. candidates got appended), but rename is still safe
//      (b) index returns non-candidate results, we attempt to map the indexed
//          ranges onto candidates in a plausible way (e.g. guess that lines
//          were inserted). If such a "near miss" is found, the rename is still
//          possible
llvm::Optional<std::vector<Range>>
adjustRenameRanges(llvm::StringRef DraftCode, llvm::StringRef Identifier,
                   std::vector<Range> Indexed, const LangOptions &LangOpts) {
  trace::Span Tracer("AdjustRenameRanges");
  assert(!Indexed.empty());
  assert(std::is_sorted(Indexed.begin(), Indexed.end()));
  std::vector<Range> Lexed =
      collectIdentifierRanges(Identifier, DraftCode, LangOpts);
  llvm::sort(Lexed);
  return getMappedRanges(Indexed, Lexed);
}

llvm::Optional<std::vector<Range>> getMappedRanges(ArrayRef<Range> Indexed,
                                                   ArrayRef<Range> Lexed) {
  trace::Span Tracer("GetMappedRanges");
  assert(!Indexed.empty());
  assert(std::is_sorted(Indexed.begin(), Indexed.end()));
  assert(std::is_sorted(Lexed.begin(), Lexed.end()));

  if (Indexed.size() > Lexed.size()) {
    vlog("The number of lexed occurrences is less than indexed occurrences");
    SPAN_ATTACH(
        Tracer, "error",
        "The number of lexed occurrences is less than indexed occurrences");
    return llvm::None;
  }
  // Fast check for the special subset case.
  if (std::includes(Indexed.begin(), Indexed.end(), Lexed.begin(), Lexed.end()))
    return Indexed.vec();

  std::vector<size_t> Best;
  size_t BestCost = std::numeric_limits<size_t>::max();
  bool HasMultiple = 0;
  std::vector<size_t> ResultStorage;
  int Fuel = 10000;
  findNearMiss(ResultStorage, Indexed, Lexed, 0, Fuel,
               [&](const std::vector<size_t> &Matched) {
                 size_t MCost =
                     renameRangeAdjustmentCost(Indexed, Lexed, Matched);
                 if (MCost < BestCost) {
                   BestCost = MCost;
                   Best = std::move(Matched);
                   HasMultiple = false; // reset
                   return;
                 }
                 if (MCost == BestCost)
                   HasMultiple = true;
               });
  if (HasMultiple) {
    vlog("The best near miss is not unique.");
    SPAN_ATTACH(Tracer, "error", "The best near miss is not unique");
    return llvm::None;
  }
  if (Best.empty()) {
    vlog("Didn't find a near miss.");
    SPAN_ATTACH(Tracer, "error", "Didn't find a near miss");
    return llvm::None;
  }
  std::vector<Range> Mapped;
  for (auto I : Best)
    Mapped.push_back(Lexed[I]);
  SPAN_ATTACH(Tracer, "mapped_ranges", static_cast<int64_t>(Mapped.size()));
  return Mapped;
}

// The cost is the sum of the implied edit sizes between successive diffs, only
// simple edits are considered:
//   - insert/remove a line (change line offset)
//   - insert/remove a character on an existing line (change column offset)
//
// Example I, total result is 1 + 1 = 2.
//   diff[0]: line + 1 <- insert a line before edit 0.
//   diff[1]: line + 1
//   diff[2]: line + 1
//   diff[3]: line + 2 <- insert a line before edits 2 and 3.
//
// Example II, total result is 1 + 1 + 1 = 3.
//   diff[0]: line + 1  <- insert a line before edit 0.
//   diff[1]: column + 1 <- remove a line between edits 0 and 1, and insert a
//   character on edit 1.
size_t renameRangeAdjustmentCost(ArrayRef<Range> Indexed, ArrayRef<Range> Lexed,
                                 ArrayRef<size_t> MappedIndex) {
  assert(Indexed.size() == MappedIndex.size());
  assert(std::is_sorted(Indexed.begin(), Indexed.end()));
  assert(std::is_sorted(Lexed.begin(), Lexed.end()));

  int LastLine = -1;
  int LastDLine = 0, LastDColumn = 0;
  int Cost = 0;
  for (size_t I = 0; I < Indexed.size(); ++I) {
    int DLine = Indexed[I].start.line - Lexed[MappedIndex[I]].start.line;
    int DColumn =
        Indexed[I].start.character - Lexed[MappedIndex[I]].start.character;
    int Line = Indexed[I].start.line;
    if (Line != LastLine)
      LastDColumn = 0; // column offsets don't carry cross lines.
    Cost += abs(DLine - LastDLine) + abs(DColumn - LastDColumn);
    std::tie(LastLine, LastDLine, LastDColumn) = std::tie(Line, DLine, DColumn);
  }
  return Cost;
}

} // namespace clangd
} // namespace clang
