//===--- IncludeCleaner.cpp - Unused/Missing Headers Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeCleaner.h"
#include "Config.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {
namespace {

/// Crawler traverses the AST and feeds in the locations of (sometimes
/// implicitly) used symbols into \p Result.
class ReferencedLocationCrawler
    : public RecursiveASTVisitor<ReferencedLocationCrawler> {
public:
  ReferencedLocationCrawler(ReferencedLocations &Result) : Result(Result) {}

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    add(DRE->getDecl());
    add(DRE->getFoundDecl());
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) {
    add(ME->getMemberDecl());
    add(ME->getFoundDecl().getDecl());
    return true;
  }

  bool VisitTagType(TagType *TT) {
    add(TT->getDecl());
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) {
    // Function definition will require redeclarations to be included.
    if (FD->isThisDeclarationADefinition())
      add(FD);
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *CCE) {
    add(CCE->getConstructor());
    return true;
  }

  bool VisitTemplateSpecializationType(TemplateSpecializationType *TST) {
    if (isNew(TST)) {
      add(TST->getTemplateName().getAsTemplateDecl()); // Primary template.
      add(TST->getAsCXXRecordDecl());                  // Specialization
    }
    return true;
  }

  bool VisitTypedefType(TypedefType *TT) {
    add(TT->getDecl());
    return true;
  }

  // Consider types of any subexpression used, even if the type is not named.
  // This is helpful in getFoo().bar(), where Foo must be complete.
  // FIXME(kirillbobyrev): Should we tweak this? It may not be desirable to
  // consider types "used" when they are not directly spelled in code.
  bool VisitExpr(Expr *E) {
    TraverseType(E->getType());
    return true;
  }

  bool TraverseType(QualType T) {
    if (isNew(T.getTypePtrOrNull())) // don't care about quals
      Base::TraverseType(T);
    return true;
  }

  bool VisitUsingDecl(UsingDecl *D) {
    for (const auto *Shadow : D->shadows())
      add(Shadow->getTargetDecl());
    return true;
  }

  // Enums may be usefully forward-declared as *complete* types by specifying
  // an underlying type. In this case, the definition should see the declaration
  // so they can be checked for compatibility.
  bool VisitEnumDecl(EnumDecl *D) {
    if (D->isThisDeclarationADefinition() && D->getIntegerTypeSourceInfo())
      add(D);
    return true;
  }

private:
  using Base = RecursiveASTVisitor<ReferencedLocationCrawler>;

  void add(const Decl *D) {
    if (!D || !isNew(D->getCanonicalDecl()))
      return;
    for (const Decl *Redecl : D->redecls())
      Result.insert(Redecl->getLocation());
  }

  bool isNew(const void *P) { return P && Visited.insert(P).second; }

  ReferencedLocations &Result;
  llvm::DenseSet<const void *> Visited;
};

// Given a set of referenced FileIDs, determines all the potentially-referenced
// files and macros by traversing expansion/spelling locations of macro IDs.
// This is used to map the referenced SourceLocations onto real files.
struct ReferencedFiles {
  ReferencedFiles(const SourceManager &SM) : SM(SM) {}
  llvm::DenseSet<FileID> Files;
  llvm::DenseSet<FileID> Macros;
  const SourceManager &SM;

  void add(SourceLocation Loc) { add(SM.getFileID(Loc), Loc); }

  void add(FileID FID, SourceLocation Loc) {
    if (FID.isInvalid())
      return;
    assert(SM.isInFileID(Loc, FID));
    if (Loc.isFileID()) {
      Files.insert(FID);
      return;
    }
    // Don't process the same macro FID twice.
    if (!Macros.insert(FID).second)
      return;
    const auto &Exp = SM.getSLocEntry(FID).getExpansion();
    add(Exp.getSpellingLoc());
    add(Exp.getExpansionLocStart());
    add(Exp.getExpansionLocEnd());
  }
};

// Returns the range starting at '#' and ending at EOL. Escaped newlines are not
// handled.
clangd::Range getDiagnosticRange(llvm::StringRef Code, unsigned HashOffset) {
  clangd::Range Result;
  Result.end = Result.start = offsetToPosition(Code, HashOffset);

  // Span the warning until the EOL or EOF.
  Result.end.character +=
      lspLength(Code.drop_front(HashOffset).take_until([](char C) {
        return C == '\n' || C == '\r';
      }));
  return Result;
}

// Finds locations of macros referenced from within the main file. That includes
// references that were not yet expanded, e.g `BAR` in `#define FOO BAR`.
void findReferencedMacros(ParsedAST &AST, ReferencedLocations &Result) {
  trace::Span Tracer("IncludeCleaner::findReferencedMacros");
  auto &SM = AST.getSourceManager();
  auto &PP = AST.getPreprocessor();
  // FIXME(kirillbobyrev): The macros from the main file are collected in
  // ParsedAST's MainFileMacros. However, we can't use it here because it
  // doesn't handle macro references that were not expanded, e.g. in macro
  // definitions or preprocessor-disabled sections.
  //
  // Extending MainFileMacros to collect missing references and switching to
  // this mechanism (as opposed to iterating through all tokens) will improve
  // the performance of findReferencedMacros and also improve other features
  // relying on MainFileMacros.
  for (const syntax::Token &Tok :
       AST.getTokens().spelledTokens(SM.getMainFileID())) {
    auto Macro = locateMacroAt(Tok, PP);
    if (!Macro)
      continue;
    auto Loc = Macro->Info->getDefinitionLoc();
    if (Loc.isValid())
      Result.insert(Loc);
  }
}

bool mayConsiderUnused(const Inclusion &Inc, ParsedAST &AST) {
  // FIXME(kirillbobyrev): We currently do not support the umbrella headers.
  // Standard Library headers are typically umbrella headers, and system
  // headers are likely to be the Standard Library headers. Until we have a
  // good support for umbrella headers and Standard Library headers, don't warn
  // about them.
  if (Inc.Written.front() == '<')
    return false;
  // Headers without include guards have side effects and are not
  // self-contained, skip them.
  assert(Inc.HeaderID);
  auto FE = AST.getSourceManager().getFileManager().getFile(
      AST.getIncludeStructure().getRealPath(
          static_cast<IncludeStructure::HeaderID>(*Inc.HeaderID)));
  assert(FE);
  if (!AST.getPreprocessor().getHeaderSearchInfo().isFileMultipleIncludeGuarded(
          *FE)) {
    dlog("{0} doesn't have header guard and will not be considered unused",
         (*FE)->getName());
    return false;
  }
  return true;
}

} // namespace

ReferencedLocations findReferencedLocations(ParsedAST &AST) {
  trace::Span Tracer("IncludeCleaner::findReferencedLocations");
  ReferencedLocations Result;
  ReferencedLocationCrawler Crawler(Result);
  Crawler.TraverseAST(AST.getASTContext());
  findReferencedMacros(AST, Result);
  return Result;
}

llvm::DenseSet<FileID>
findReferencedFiles(const llvm::DenseSet<SourceLocation> &Locs,
                    const SourceManager &SM) {
  std::vector<SourceLocation> Sorted{Locs.begin(), Locs.end()};
  llvm::sort(Sorted); // Group by FileID.
  ReferencedFiles Result(SM);
  for (auto It = Sorted.begin(); It < Sorted.end();) {
    FileID FID = SM.getFileID(*It);
    Result.add(FID, *It);
    // Cheaply skip over all the other locations from the same FileID.
    // This avoids lots of redundant Loc->File lookups for the same file.
    do
      ++It;
    while (It != Sorted.end() && SM.isInFileID(*It, FID));
  }
  return std::move(Result.Files);
}

std::vector<const Inclusion *>
getUnused(ParsedAST &AST,
          const llvm::DenseSet<IncludeStructure::HeaderID> &ReferencedFiles) {
  trace::Span Tracer("IncludeCleaner::getUnused");
  std::vector<const Inclusion *> Unused;
  for (const Inclusion &MFI : AST.getIncludeStructure().MainFileIncludes) {
    if (!MFI.HeaderID)
      continue;
    auto IncludeID = static_cast<IncludeStructure::HeaderID>(*MFI.HeaderID);
    bool Used = ReferencedFiles.contains(IncludeID);
    if (!Used && !mayConsiderUnused(MFI, AST)) {
      dlog("{0} was not used, but is not eligible to be diagnosed as unused",
           MFI.Written);
      continue;
    }
    if (!Used)
      Unused.push_back(&MFI);
    dlog("{0} is {1}", MFI.Written, Used ? "USED" : "UNUSED");
  }
  return Unused;
}

#ifndef NDEBUG
// Is FID a <built-in>, <scratch space> etc?
static bool isSpecialBuffer(FileID FID, const SourceManager &SM) {
  const SrcMgr::FileInfo &FI = SM.getSLocEntry(FID).getFile();
  return FI.getName().startswith("<");
}
#endif

llvm::DenseSet<IncludeStructure::HeaderID>
translateToHeaderIDs(const llvm::DenseSet<FileID> &Files,
                     const IncludeStructure &Includes,
                     const SourceManager &SM) {
  trace::Span Tracer("IncludeCleaner::translateToHeaderIDs");
  llvm::DenseSet<IncludeStructure::HeaderID> TranslatedHeaderIDs;
  TranslatedHeaderIDs.reserve(Files.size());
  for (FileID FID : Files) {
    const FileEntry *FE = SM.getFileEntryForID(FID);
    if (!FE) {
      assert(isSpecialBuffer(FID, SM));
      continue;
    }
    const auto File = Includes.getID(FE);
    assert(File);
    TranslatedHeaderIDs.insert(*File);
  }
  return TranslatedHeaderIDs;
}

std::vector<const Inclusion *> computeUnusedIncludes(ParsedAST &AST) {
  const auto &SM = AST.getSourceManager();

  auto Refs = findReferencedLocations(AST);
  // FIXME(kirillbobyrev): Attribute the symbols from non self-contained
  // headers to their parents while the information about respective
  // SourceLocations and FileIDs is not lost. It is not possible to do that
  // later when non-guarded headers included multiple times will get same
  // HeaderID.
  auto ReferencedFileIDs = findReferencedFiles(Refs, SM);
  auto ReferencedHeaders =
      translateToHeaderIDs(ReferencedFileIDs, AST.getIncludeStructure(), SM);
  return getUnused(AST, ReferencedHeaders);
}

std::vector<Diag> issueUnusedIncludesDiagnostics(ParsedAST &AST,
                                                 llvm::StringRef Code) {
  trace::Span Tracer("IncludeCleaner::issueUnusedIncludesDiagnostics");
  const Config &Cfg = Config::current();
  if (Cfg.Diagnostics.UnusedIncludes != Config::UnusedIncludesPolicy::Strict ||
      Cfg.Diagnostics.SuppressAll ||
      Cfg.Diagnostics.Suppress.contains("unused-includes"))
    return {};
  std::vector<Diag> Result;
  std::string FileName =
      AST.getSourceManager()
          .getFileEntryForID(AST.getSourceManager().getMainFileID())
          ->getName()
          .str();
  for (const auto *Inc : computeUnusedIncludes(AST)) {
    Diag D;
    D.Message =
        llvm::formatv("included header {0} is not used",
                      llvm::sys::path::filename(
                          Inc->Written.substr(1, Inc->Written.size() - 2),
                          llvm::sys::path::Style::posix));
    D.Name = "unused-includes";
    D.Source = Diag::DiagSource::Clangd;
    D.File = FileName;
    D.Severity = DiagnosticsEngine::Warning;
    D.Tags.push_back(Unnecessary);
    D.Range = getDiagnosticRange(Code, Inc->HashOffset);
    // FIXME(kirillbobyrev): Removing inclusion might break the code if the
    // used headers are only reachable transitively through this one. Suggest
    // including them directly instead.
    // FIXME(kirillbobyrev): Add fix suggestion for adding IWYU pragmas
    // (keep/export) remove the warning once we support IWYU pragmas.
    D.Fixes.emplace_back();
    D.Fixes.back().Message = "remove #include directive";
    D.Fixes.back().Edits.emplace_back();
    D.Fixes.back().Edits.back().range.start.line = Inc->HashLine;
    D.Fixes.back().Edits.back().range.end.line = Inc->HashLine + 1;
    D.InsideMainFile = true;
    Result.push_back(std::move(D));
  }
  return Result;
}

} // namespace clangd
} // namespace clang
