//===--- Rename.cpp - Symbol-rename refactorings -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/Rename.h"
#include "AST.h"
#include "Logger.h"
#include "index/SymbolCollector.h"
#include "clang/Tooling/Refactoring/RefactoringResultConsumer.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"

namespace clang {
namespace clangd {
namespace {

class RefactoringResultCollector final
    : public tooling::RefactoringResultConsumer {
public:
  void handleError(llvm::Error Err) override {
    assert(!Result.hasValue());
    Result = std::move(Err);
  }

  // Using the handle(SymbolOccurrences) from parent class.
  using tooling::RefactoringResultConsumer::handle;

  void handle(tooling::AtomicChanges SourceReplacements) override {
    assert(!Result.hasValue());
    Result = std::move(SourceReplacements);
  }

  llvm::Optional<llvm::Expected<tooling::AtomicChanges>> Result;
};

// Expand a DiagnosticError to make it print-friendly (print the detailed
// message, rather than "clang diagnostic").
llvm::Error expandDiagnostics(llvm::Error Err, DiagnosticsEngine &DE) {
  if (auto Diag = DiagnosticError::take(Err)) {
    llvm::cantFail(std::move(Err));
    SmallVector<char, 128> DiagMessage;
    Diag->second.EmitToString(DE, DiagMessage);
    return llvm::make_error<llvm::StringError>(DiagMessage,
                                               llvm::inconvertibleErrorCode());
  }
  return Err;
}

llvm::Optional<std::string> filePath(const SymbolLocation &Loc,
                                     llvm::StringRef HintFilePath) {
  if (!Loc)
    return None;
  auto Uri = URI::parse(Loc.FileURI);
  if (!Uri) {
    elog("Could not parse URI {0}: {1}", Loc.FileURI, Uri.takeError());
    return None;
  }
  auto U = URIForFile::fromURI(*Uri, HintFilePath);
  if (!U) {
    elog("Could not resolve URI {0}: {1}", Loc.FileURI, U.takeError());
    return None;
  }
  return U->file().str();
}

// Query the index to find some other files where the Decl is referenced.
llvm::Optional<std::string> getOtherRefFile(const Decl &D, StringRef MainFile,
                                            const SymbolIndex &Index) {
  RefsRequest Req;
  // We limit the number of results, this is a correctness/performance
  // tradeoff. We expect the number of symbol references in the current file
  // is smaller than the limit.
  Req.Limit = 100;
  if (auto ID = getSymbolID(&D))
    Req.IDs.insert(*ID);
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

enum ReasonToReject {
  NoIndexProvided,
  NonIndexable,
  UsedOutsideFile,
  UnsupportedSymbol,
};

// Check the symbol Decl is renameable (per the index) within the file.
llvm::Optional<ReasonToReject> renamableWithinFile(const Decl &RenameDecl,
                                                   StringRef MainFile,
                                                   const SymbolIndex *Index) {
  if (llvm::isa<NamespaceDecl>(&RenameDecl))
    return ReasonToReject::UnsupportedSymbol;
  auto &ASTCtx = RenameDecl.getASTContext();
  const auto &SM = ASTCtx.getSourceManager();
  bool MainFileIsHeader = ASTCtx.getLangOpts().IsHeaderFile;
  bool DeclaredInMainFile = isInsideMainFile(RenameDecl.getBeginLoc(), SM);

  // If the symbol is declared in the main file (which is not a header), we
  // rename it.
  if (DeclaredInMainFile && !MainFileIsHeader)
    return None;

  // Below are cases where the symbol is declared in the header.
  // If the symbol is function-local, we rename it.
  if (RenameDecl.getParentFunctionOrMethod())
    return None;

  if (!Index)
    return ReasonToReject::NoIndexProvided;

  bool IsIndexable = isa<NamedDecl>(RenameDecl) &&
                     SymbolCollector::shouldCollectSymbol(
                         cast<NamedDecl>(RenameDecl), ASTCtx, {}, false);
  // If the symbol is not indexable, we disallow rename.
  if (!IsIndexable)
    return ReasonToReject::NonIndexable;
  auto OtherFile = getOtherRefFile(RenameDecl, MainFile, *Index);
  // If the symbol is indexable and has no refs from other files in the index,
  // we rename it.
  if (!OtherFile)
    return None;
  // If the symbol is indexable and has refs from other files in the index,
  // we disallow rename.
  return ReasonToReject::UsedOutsideFile;
}

llvm::Error makeError(ReasonToReject Reason) {
  auto Message = [](ReasonToReject Reason) {
    switch (Reason) {
    case NoIndexProvided:
      return "symbol may be used in other files (no index available)";
    case UsedOutsideFile:
      return "the symbol is used outside main file";
    case NonIndexable:
      return "symbol may be used in other files (not eligible for indexing)";
    case UnsupportedSymbol:
      return "symbol is not a supported kind (e.g. namespace, macro)";
    }
    llvm_unreachable("unhandled reason kind");
  };
  return llvm::make_error<llvm::StringError>(
      llvm::formatv("Cannot rename symbol: {0}", Message(Reason)),
      llvm::inconvertibleErrorCode());
}

} // namespace

llvm::Expected<tooling::Replacements>
renameWithinFile(ParsedAST &AST, llvm::StringRef File, Position Pos,
                 llvm::StringRef NewName, const SymbolIndex *Index) {
  RefactoringResultCollector ResultCollector;
  ASTContext &ASTCtx = AST.getASTContext();
  SourceLocation SourceLocationBeg = clangd::getBeginningOfIdentifier(
      AST, Pos, AST.getSourceManager().getMainFileID());
  // FIXME: renaming macros is not supported yet, the macro-handling code should
  // be moved to rename tooling library.
  if (locateMacroAt(SourceLocationBeg, AST.getPreprocessor()))
    return makeError(UnsupportedSymbol);
  tooling::RefactoringRuleContext Context(AST.getSourceManager());
  Context.setASTContext(ASTCtx);
  auto Rename = clang::tooling::RenameOccurrences::initiate(
      Context, SourceRange(SourceLocationBeg), NewName);
  if (!Rename)
    return expandDiagnostics(Rename.takeError(), ASTCtx.getDiagnostics());

  const auto *RenameDecl = Rename->getRenameDecl();
  assert(RenameDecl && "symbol must be found at this point");
  if (auto Reject =
          renamableWithinFile(*RenameDecl->getCanonicalDecl(), File, Index))
    return makeError(*Reject);

  Rename->invoke(ResultCollector, Context);

  assert(ResultCollector.Result.hasValue());
  if (!ResultCollector.Result.getValue())
    return expandDiagnostics(ResultCollector.Result->takeError(),
                             ASTCtx.getDiagnostics());

  tooling::Replacements FilteredChanges;
  // Right now we only support renaming the main file, so we
  // drop replacements not for the main file. In the future, we might
  // also support rename with wider scope.
  // Rename sometimes returns duplicate edits (which is a bug). A side-effect of
  // adding them to a single Replacements object is these are deduplicated.
  for (const tooling::AtomicChange &Change : ResultCollector.Result->get()) {
    for (const auto &Rep : Change.getReplacements()) {
      if (Rep.getFilePath() == File)
        cantFail(FilteredChanges.add(Rep));
    }
  }
  return FilteredChanges;
}

} // namespace clangd
} // namespace clang
