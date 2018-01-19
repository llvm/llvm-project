//===--- SymbolCollector.cpp -------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolCollector.h"
#include "../CodeCompletionStrings.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {

namespace {
// Make the Path absolute using the current working directory of the given
// SourceManager if the Path is not an absolute path.
//
// The Path can be a path relative to the build directory, or retrieved from
// the SourceManager.
std::string makeAbsolutePath(const SourceManager &SM, StringRef Path) {
  llvm::SmallString<128> AbsolutePath(Path);
  if (std::error_code EC =
          SM.getFileManager().getVirtualFileSystem()->makeAbsolute(
              AbsolutePath))
    llvm::errs() << "Warning: could not make absolute file: '" << EC.message()
                 << '\n';
  // Handle the symbolic link path case where the current working directory
  // (getCurrentWorkingDirectory) is a symlink./ We always want to the real
  // file path (instead of the symlink path) for the  C++ symbols.
  //
  // Consider the following example:
  //
  //   src dir: /project/src/foo.h
  //   current working directory (symlink): /tmp/build -> /project/src/
  //
  // The file path of Symbol is "/project/src/foo.h" instead of
  // "/tmp/build/foo.h"
  const DirectoryEntry *Dir = SM.getFileManager().getDirectory(
      llvm::sys::path::parent_path(AbsolutePath.str()));
  if (Dir) {
    StringRef DirName = SM.getFileManager().getCanonicalName(Dir);
    SmallString<128> AbsoluteFilename;
    llvm::sys::path::append(AbsoluteFilename, DirName,
                            llvm::sys::path::filename(AbsolutePath.str()));
    return AbsoluteFilename.str();
  }
  return AbsolutePath.str();
}

// "a::b::c", return {"a::b::", "c"}. Scope is empty if there's no qualifier.
std::pair<llvm::StringRef, llvm::StringRef>
splitQualifiedName(llvm::StringRef QName) {
  assert(!QName.startswith("::") && "Qualified names should not start with ::");
  size_t Pos = QName.rfind("::");
  if (Pos == llvm::StringRef::npos)
    return {StringRef(), QName};
  return {QName.substr(0, Pos + 2), QName.substr(Pos + 2)};
}

bool shouldFilterDecl(const NamedDecl *ND, ASTContext *ASTCtx,
                      const SymbolCollector::Options &Opts) {
  using namespace clang::ast_matchers;
  if (ND->isImplicit())
    return true;
  // Skip anonymous declarations, e.g (anonymous enum/class/struct).
  if (ND->getDeclName().isEmpty())
    return true;

  // FIXME: figure out a way to handle internal linkage symbols (e.g. static
  // variables, function) defined in the .cc files. Also we skip the symbols
  // in anonymous namespace as the qualifier names of these symbols are like
  // `foo::<anonymous>::bar`, which need a special handling.
  // In real world projects, we have a relatively large set of header files
  // that define static variables (like "static const int A = 1;"), we still
  // want to collect these symbols, although they cause potential ODR
  // violations.
  if (ND->isInAnonymousNamespace())
    return true;

  // We only want:
  //   * symbols in namespaces or translation unit scopes (e.g. no class
  //     members)
  //   * enum constants in unscoped enum decl (e.g. "red" in "enum {red};")
  auto InTopLevelScope =
      hasDeclContext(anyOf(namespaceDecl(), translationUnitDecl()));
  if (match(decl(allOf(Opts.IndexMainFiles
                           ? decl()
                           : decl(unless(isExpansionInMainFile())),
                       anyOf(InTopLevelScope,
                             hasDeclContext(enumDecl(InTopLevelScope,
                                                     unless(isScoped())))))),
            *ND, *ASTCtx)
          .empty())
    return true;

  return false;
}

} // namespace

SymbolCollector::SymbolCollector(Options Opts) : Opts(std::move(Opts)) {}

void SymbolCollector::initialize(ASTContext &Ctx) {
  ASTCtx = &Ctx;
  CompletionAllocator = std::make_shared<GlobalCodeCompletionAllocator>();
  CompletionTUInfo =
      llvm::make_unique<CodeCompletionTUInfo>(CompletionAllocator);
}

// Always return true to continue indexing.
bool SymbolCollector::handleDeclOccurence(
    const Decl *D, index::SymbolRoleSet Roles,
    ArrayRef<index::SymbolRelation> Relations, FileID FID, unsigned Offset,
    index::IndexDataConsumer::ASTNodeInfo ASTNode) {
  assert(ASTCtx && PP.get() && "ASTContext and Preprocessor must be set.");

  // FIXME: collect all symbol references.
  if (!(Roles & static_cast<unsigned>(index::SymbolRole::Declaration) ||
        Roles & static_cast<unsigned>(index::SymbolRole::Definition)))
    return true;

  assert(CompletionAllocator && CompletionTUInfo);

  if (const NamedDecl *ND = llvm::dyn_cast<NamedDecl>(D)) {
    if (shouldFilterDecl(ND, ASTCtx, Opts))
      return true;
    llvm::SmallString<128> USR;
    if (index::generateUSRForDecl(ND, USR))
      return true;

    auto ID = SymbolID(USR);
    if (Symbols.find(ID) != nullptr)
      return true;

    auto &SM = ND->getASTContext().getSourceManager();
    std::string FilePath =
        makeAbsolutePath(SM, SM.getFilename(D->getLocation()));
    SymbolLocation Location = {FilePath, SM.getFileOffset(D->getLocStart()),
                               SM.getFileOffset(D->getLocEnd())};
    std::string QName = ND->getQualifiedNameAsString();

    Symbol S;
    S.ID = std::move(ID);
    std::tie(S.Scope, S.Name) = splitQualifiedName(QName);
    S.SymInfo = index::getSymbolInfo(D);
    S.CanonicalDeclaration = Location;

    // Add completion info.
    assert(ASTCtx && PP.get() && "ASTContext and Preprocessor must be set.");
    CodeCompletionResult SymbolCompletion(ND, 0);
    const auto *CCS = SymbolCompletion.CreateCodeCompletionString(
        *ASTCtx, *PP, CodeCompletionContext::CCC_Name, *CompletionAllocator,
        *CompletionTUInfo,
        /*IncludeBriefComments*/ true);
    std::string Label;
    std::string SnippetInsertText;
    std::string IgnoredLabel;
    std::string PlainInsertText;
    getLabelAndInsertText(*CCS, &Label, &SnippetInsertText,
                          /*EnableSnippets=*/true);
    getLabelAndInsertText(*CCS, &IgnoredLabel, &PlainInsertText,
                          /*EnableSnippets=*/false);
    std::string FilterText = getFilterText(*CCS);
    std::string Documentation = getDocumentation(*CCS);
    std::string CompletionDetail = getDetail(*CCS);

    S.CompletionFilterText = FilterText;
    S.CompletionLabel = Label;
    S.CompletionPlainInsertText = PlainInsertText;
    S.CompletionSnippetInsertText = SnippetInsertText;
    Symbol::Details Detail;
    Detail.Documentation = Documentation;
    Detail.CompletionDetail = CompletionDetail;
    S.Detail = &Detail;

    Symbols.insert(S);
  }

  return true;
}

} // namespace clangd
} // namespace clang
