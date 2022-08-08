//===--- IncludeCleaner.cpp - Unused/Missing Headers Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeCleaner.h"
#include "Config.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "index/CanonicalIncludes.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include <functional>

namespace clang {
namespace clangd {

static bool AnalyzeStdlib = false;
void setIncludeCleanerAnalyzesStdlib(bool B) { AnalyzeStdlib = B; }

namespace {

/// Crawler traverses the AST and feeds in the locations of (sometimes
/// implicitly) used symbols into \p Result.
class ReferencedLocationCrawler
    : public RecursiveASTVisitor<ReferencedLocationCrawler> {
public:
  ReferencedLocationCrawler(ReferencedLocations &Result,
                            const SourceManager &SM)
      : Result(Result), SM(SM) {}

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
    // Using templateName case is handled by the override TraverseTemplateName.
    if (TST->getTemplateName().getKind() == TemplateName::UsingTemplate)
      return true;
    add(TST->getAsCXXRecordDecl());                  // Specialization
    return true;
  }

  // There is no VisitTemplateName in RAV, thus we override the Traverse version
  // to handle the Using TemplateName case.
  bool TraverseTemplateName(TemplateName TN) {
    VisitTemplateName(TN);
    return Base::TraverseTemplateName(TN);
  }
  // A pseudo VisitTemplateName, dispatched by the above TraverseTemplateName!
  bool VisitTemplateName(TemplateName TN) {
    if (const auto *USD = TN.getAsUsingShadowDecl()) {
      add(USD);
      return true;
    }
    add(TN.getAsTemplateDecl()); // Primary template.
    return true;
  }

  bool VisitUsingType(UsingType *UT) {
    add(UT->getFoundDecl());
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

  // When the overload is not resolved yet, mark all candidates as used.
  bool VisitOverloadExpr(OverloadExpr *E) {
    for (const auto *ResolutionDecl : E->decls())
      add(ResolutionDecl);
    return true;
  }

private:
  using Base = RecursiveASTVisitor<ReferencedLocationCrawler>;

  void add(const Decl *D) {
    if (!D || !isNew(D->getCanonicalDecl()))
      return;
    if (auto SS = StdRecognizer(D)) {
      Result.Stdlib.insert(*SS);
      return;
    }
    // Special case RecordDecls, as it is common for them to be forward
    // declared multiple times. The most common cases are:
    // - Definition available in TU, only mark that one as usage. The rest is
    //   likely to be unnecessary. This might result in false positives when an
    //   internal definition is visible.
    // - There's a forward declaration in the main file, no need for other
    //   redecls.
    if (const auto *RD = llvm::dyn_cast<RecordDecl>(D)) {
      if (const auto *Definition = RD->getDefinition()) {
        Result.User.insert(Definition->getLocation());
        return;
      }
      if (SM.isInMainFile(RD->getMostRecentDecl()->getLocation()))
        return;
    }
    for (const Decl *Redecl : D->redecls())
      Result.User.insert(Redecl->getLocation());
  }

  bool isNew(const void *P) { return P && Visited.insert(P).second; }

  ReferencedLocations &Result;
  llvm::DenseSet<const void *> Visited;
  const SourceManager &SM;
  tooling::stdlib::Recognizer StdRecognizer;
};

// Given a set of referenced FileIDs, determines all the potentially-referenced
// files and macros by traversing expansion/spelling locations of macro IDs.
// This is used to map the referenced SourceLocations onto real files.
struct ReferencedFilesBuilder {
  ReferencedFilesBuilder(const SourceManager &SM) : SM(SM) {}
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
void findReferencedMacros(const SourceManager &SM, Preprocessor &PP,
                          const syntax::TokenBuffer *Tokens,
                          ReferencedLocations &Result) {
  trace::Span Tracer("IncludeCleaner::findReferencedMacros");
  // FIXME(kirillbobyrev): The macros from the main file are collected in
  // ParsedAST's MainFileMacros. However, we can't use it here because it
  // doesn't handle macro references that were not expanded, e.g. in macro
  // definitions or preprocessor-disabled sections.
  //
  // Extending MainFileMacros to collect missing references and switching to
  // this mechanism (as opposed to iterating through all tokens) will improve
  // the performance of findReferencedMacros and also improve other features
  // relying on MainFileMacros.
  for (const syntax::Token &Tok : Tokens->spelledTokens(SM.getMainFileID())) {
    auto Macro = locateMacroAt(Tok, PP);
    if (!Macro)
      continue;
    auto Loc = Macro->Info->getDefinitionLoc();
    if (Loc.isValid())
      Result.User.insert(Loc);
    // FIXME: support stdlib macros
  }
}

static bool mayConsiderUnused(const Inclusion &Inc, ParsedAST &AST,
                              const Config &Cfg) {
  if (Inc.BehindPragmaKeep)
    return false;

  // FIXME(kirillbobyrev): We currently do not support the umbrella headers.
  // System headers are likely to be standard library headers.
  // Until we have good support for umbrella headers, don't warn about them.
  if (Inc.Written.front() == '<') {
    if (AnalyzeStdlib && tooling::stdlib::Header::named(Inc.Written))
      return true;
    return false;
  }
  assert(Inc.HeaderID);
  auto HID = static_cast<IncludeStructure::HeaderID>(*Inc.HeaderID);
  // FIXME: Ignore the headers with IWYU export pragmas for now, remove this
  // check when we have more precise tracking of exported headers.
  if (AST.getIncludeStructure().hasIWYUExport(HID))
    return false;
  auto FE = AST.getSourceManager().getFileManager().getFileRef(
      AST.getIncludeStructure().getRealPath(HID));
  assert(FE);
  // Headers without include guards have side effects and are not
  // self-contained, skip them.
  if (!AST.getPreprocessor().getHeaderSearchInfo().isFileMultipleIncludeGuarded(
          &FE->getFileEntry())) {
    dlog("{0} doesn't have header guard and will not be considered unused",
         FE->getName());
    return false;
  }
  for (auto &Filter : Cfg.Diagnostics.Includes.IgnoreHeader) {
    // Convert the path to Unix slashes and try to match against the filter.
    llvm::SmallString<64> Path(Inc.Resolved);
    llvm::sys::path::native(Path, llvm::sys::path::Style::posix);
    if (Filter(Inc.Resolved)) {
      dlog("{0} header is filtered out by the configuration", FE->getName());
      return false;
    }
  }
  return true;
}

// In case symbols are coming from non self-contained header, we need to find
// its first includer that is self-contained. This is the header users can
// include, so it will be responsible for bringing the symbols from given
// header into the scope.
FileID headerResponsible(FileID ID, const SourceManager &SM,
                         const IncludeStructure &Includes) {
  // Unroll the chain of non self-contained headers until we find the one that
  // can be included.
  for (const FileEntry *FE = SM.getFileEntryForID(ID); ID != SM.getMainFileID();
       FE = SM.getFileEntryForID(ID)) {
    // If FE is nullptr, we consider it to be the responsible header.
    if (!FE)
      break;
    auto HID = Includes.getID(FE);
    assert(HID && "We're iterating over headers already existing in "
                  "IncludeStructure");
    if (Includes.isSelfContained(*HID))
      break;
    // The header is not self-contained: put the responsibility for its symbols
    // on its includer.
    ID = SM.getFileID(SM.getIncludeLoc(ID));
  }
  return ID;
}

} // namespace

ReferencedLocations findReferencedLocations(ASTContext &Ctx, Preprocessor &PP,
                                            const syntax::TokenBuffer *Tokens) {
  trace::Span Tracer("IncludeCleaner::findReferencedLocations");
  ReferencedLocations Result;
  const auto &SM = Ctx.getSourceManager();
  ReferencedLocationCrawler Crawler(Result, SM);
  Crawler.TraverseAST(Ctx);
  if (Tokens)
    findReferencedMacros(SM, PP, Tokens, Result);
  return Result;
}

ReferencedLocations findReferencedLocations(ParsedAST &AST) {
  return findReferencedLocations(AST.getASTContext(), AST.getPreprocessor(),
                                 &AST.getTokens());
}

ReferencedFiles findReferencedFiles(
    const ReferencedLocations &Locs, const SourceManager &SM,
    llvm::function_ref<FileID(FileID)> HeaderResponsible,
    llvm::function_ref<Optional<StringRef>(FileID)> UmbrellaHeader) {
  std::vector<SourceLocation> Sorted{Locs.User.begin(), Locs.User.end()};
  llvm::sort(Sorted); // Group by FileID.
  ReferencedFilesBuilder Builder(SM);
  for (auto It = Sorted.begin(); It < Sorted.end();) {
    FileID FID = SM.getFileID(*It);
    Builder.add(FID, *It);
    // Cheaply skip over all the other locations from the same FileID.
    // This avoids lots of redundant Loc->File lookups for the same file.
    do
      ++It;
    while (It != Sorted.end() && SM.isInFileID(*It, FID));
  }

  // If a header is not self-contained, we consider its symbols a logical part
  // of the including file. Therefore, mark the parents of all used
  // non-self-contained FileIDs as used. Perform this on FileIDs rather than
  // HeaderIDs, as each inclusion of a non-self-contained file is distinct.
  llvm::DenseSet<FileID> UserFiles;
  llvm::StringSet<> PublicHeaders;
  for (FileID ID : Builder.Files) {
    UserFiles.insert(HeaderResponsible(ID));
    if (auto PublicHeader = UmbrellaHeader(ID)) {
      PublicHeaders.insert(*PublicHeader);
    }
  }

  llvm::DenseSet<tooling::stdlib::Header> StdlibFiles;
  for (const auto &Symbol : Locs.Stdlib)
    for (const auto &Header : Symbol.headers())
      StdlibFiles.insert(Header);

  return {std::move(UserFiles), std::move(StdlibFiles),
          std::move(PublicHeaders)};
}

ReferencedFiles findReferencedFiles(const ReferencedLocations &Locs,
                                    const IncludeStructure &Includes,
                                    const CanonicalIncludes &CanonIncludes,
                                    const SourceManager &SM) {
  return findReferencedFiles(
      Locs, SM,
      [&SM, &Includes](FileID ID) {
        return headerResponsible(ID, SM, Includes);
      },
      [&SM, &CanonIncludes](FileID ID) -> Optional<StringRef> {
        auto Entry = SM.getFileEntryRefForID(ID);
        if (!Entry)
          return llvm::None;
        auto PublicHeader = CanonIncludes.mapHeader(*Entry);
        if (PublicHeader.empty())
          return llvm::None;
        return PublicHeader;
      });
}

std::vector<const Inclusion *>
getUnused(ParsedAST &AST,
          const llvm::DenseSet<IncludeStructure::HeaderID> &ReferencedFiles,
          const llvm::StringSet<> &ReferencedPublicHeaders) {
  trace::Span Tracer("IncludeCleaner::getUnused");
  const Config &Cfg = Config::current();
  std::vector<const Inclusion *> Unused;
  for (const Inclusion &MFI : AST.getIncludeStructure().MainFileIncludes) {
    if (!MFI.HeaderID)
      continue;
    if (ReferencedPublicHeaders.contains(MFI.Written))
      continue;
    auto IncludeID = static_cast<IncludeStructure::HeaderID>(*MFI.HeaderID);
    bool Used = ReferencedFiles.contains(IncludeID);
    if (!Used && !mayConsiderUnused(MFI, AST, Cfg)) {
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
translateToHeaderIDs(const ReferencedFiles &Files,
                     const IncludeStructure &Includes,
                     const SourceManager &SM) {
  trace::Span Tracer("IncludeCleaner::translateToHeaderIDs");
  llvm::DenseSet<IncludeStructure::HeaderID> TranslatedHeaderIDs;
  TranslatedHeaderIDs.reserve(Files.User.size());
  for (FileID FID : Files.User) {
    const FileEntry *FE = SM.getFileEntryForID(FID);
    if (!FE) {
      assert(isSpecialBuffer(FID, SM));
      continue;
    }
    const auto File = Includes.getID(FE);
    assert(File);
    TranslatedHeaderIDs.insert(*File);
  }
  for (tooling::stdlib::Header StdlibUsed : Files.Stdlib)
    for (auto HID : Includes.StdlibHeaders.lookup(StdlibUsed))
      TranslatedHeaderIDs.insert(HID);
  return TranslatedHeaderIDs;
}

std::vector<const Inclusion *> computeUnusedIncludes(ParsedAST &AST) {
  const auto &SM = AST.getSourceManager();

  auto Refs = findReferencedLocations(AST);
  auto ReferencedFiles =
      findReferencedFiles(Refs, AST.getIncludeStructure(),
                          AST.getCanonicalIncludes(), AST.getSourceManager());
  auto ReferencedHeaders =
      translateToHeaderIDs(ReferencedFiles, AST.getIncludeStructure(), SM);
  return getUnused(AST, ReferencedHeaders, ReferencedFiles.SpelledUmbrellas);
}

std::vector<Diag> issueUnusedIncludesDiagnostics(ParsedAST &AST,
                                                 llvm::StringRef Code) {
  const Config &Cfg = Config::current();
  if (Cfg.Diagnostics.UnusedIncludes != Config::UnusedIncludesPolicy::Strict ||
      Cfg.Diagnostics.SuppressAll ||
      Cfg.Diagnostics.Suppress.contains("unused-includes"))
    return {};
  trace::Span Tracer("IncludeCleaner::issueUnusedIncludesDiagnostics");
  std::vector<Diag> Result;
  std::string FileName =
      AST.getSourceManager()
          .getFileEntryRefForID(AST.getSourceManager().getMainFileID())
          ->getName()
          .str();
  for (const auto *Inc : computeUnusedIncludes(AST)) {
    Diag D;
    D.Message =
        llvm::formatv("included header {0} is not used directly",
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
