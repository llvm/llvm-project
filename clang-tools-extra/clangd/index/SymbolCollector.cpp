//===--- SymbolCollector.cpp -------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolCollector.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
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
    SmallVector<char, 128> AbsoluteFilename;
    llvm::sys::path::append(AbsoluteFilename, DirName,
                            llvm::sys::path::filename(AbsolutePath.str()));
    return llvm::StringRef(AbsoluteFilename.data(), AbsoluteFilename.size())
        .str();
  }
  return AbsolutePath.str();
}

// Split a qualified symbol name into scope and unqualified name, e.g. given
// "a::b::c", return {"a::b", "c"}. Scope is empty if it doesn't exist.
std::pair<llvm::StringRef, llvm::StringRef>
splitQualifiedName(llvm::StringRef QName) {
  assert(!QName.startswith("::") && "Qualified names should not start with ::");
  size_t Pos = QName.rfind("::");
  if (Pos == llvm::StringRef::npos)
    return {StringRef(), QName};
  return {QName.substr(0, Pos), QName.substr(Pos + 2)};
}

} // namespace

// Always return true to continue indexing.
bool SymbolCollector::handleDeclOccurence(
    const Decl *D, index::SymbolRoleSet Roles,
    ArrayRef<index::SymbolRelation> Relations, FileID FID, unsigned Offset,
    index::IndexDataConsumer::ASTNodeInfo ASTNode) {
  // FIXME: collect all symbol references.
  if (!(Roles & static_cast<unsigned>(index::SymbolRole::Declaration) ||
        Roles & static_cast<unsigned>(index::SymbolRole::Definition)))
    return true;

  if (const NamedDecl *ND = llvm::dyn_cast<NamedDecl>(D)) {
    // FIXME: Should we include the internal linkage symbols?
    if (!ND->hasExternalFormalLinkage() || ND->isInAnonymousNamespace())
      return true;

    llvm::SmallVector<char, 128> Buff;
    if (index::generateUSRForDecl(ND, Buff))
      return true;

    std::string USR(Buff.data(), Buff.size());
    auto ID = SymbolID(USR);
    if (Symbols.find(ID) != Symbols.end())
      return true;

    auto &SM = ND->getASTContext().getSourceManager();
    std::string FilePath =
        makeAbsolutePath(SM, SM.getFilename(D->getLocation()));
    SymbolLocation Location = {FilePath, SM.getFileOffset(D->getLocStart()),
                               SM.getFileOffset(D->getLocEnd())};
    std::string QName = ND->getQualifiedNameAsString();
    auto ScopeAndName = splitQualifiedName(QName);

    Symbol S;
    S.ID = std::move(ID);
    S.Scope = ScopeAndName.first;
    S.Name = ScopeAndName.second;
    S.SymInfo = index::getSymbolInfo(D);
    S.CanonicalDeclaration = Location;
    Symbols.insert(S);
  }

  return true;
}

void SymbolCollector::finish() { Symbols.freeze(); }

} // namespace clangd
} // namespace clang
