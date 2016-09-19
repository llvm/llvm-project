//===-- ChangeNamespace.h -- Change namespace  ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CHANGE_NAMESPACE_CHANGENAMESPACE_H
#define LLVM_CLANG_TOOLS_EXTRA_CHANGE_NAMESPACE_CHANGENAMESPACE_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include <string>

namespace clang {
namespace change_namespace {

// This tool can be used to change the surrounding namespaces of class/function
// definitions. Classes/functions in the moved namespace will have new
// namespaces while references to symbols (e.g. types, functions) which are not
// defined in the changed namespace will be correctly qualified by prepending
// namespace specifiers before them.
// For classes, only classes that are declared/defined in the given namespace in
// speficifed files will be moved: forward declarations will remain in the old
// namespace.
// For example, changing "a" to "x":
// Old code:
//   namespace a {
//   class FWD;
//   class A { FWD *fwd; }
//   }  // a
// New code:
//   namespace a {
//   class FWD;
//   }  // a
//   namespace x {
//   class A { a::FWD *fwd; }
//   }  // x
// FIXME: support moving typedef, enums across namespaces.
class ChangeNamespaceTool : public ast_matchers::MatchFinder::MatchCallback {
public:
  // Moves code in the old namespace `OldNs` to the new namespace `NewNs` in
  // files matching `FilePattern`.
  ChangeNamespaceTool(
      llvm::StringRef OldNs, llvm::StringRef NewNs, llvm::StringRef FilePattern,
      std::map<std::string, tooling::Replacements> *FileToReplacements,
      llvm::StringRef FallbackStyle = "LLVM");

  void registerMatchers(ast_matchers::MatchFinder *Finder);

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  // Moves the changed code in old namespaces but leaves class forward
  // declarations behind.
  void onEndOfTranslationUnit() override;

private:
  void moveOldNamespace(const ast_matchers::MatchFinder::MatchResult &Result,
                        const NamespaceDecl *NsDecl);

  void moveClassForwardDeclaration(
      const ast_matchers::MatchFinder::MatchResult &Result,
      const CXXRecordDecl *FwdDecl);

  void replaceQualifiedSymbolInDeclContext(
      const ast_matchers::MatchFinder::MatchResult &Result,
      const Decl *DeclContext, SourceLocation Start, SourceLocation End,
      llvm::StringRef DeclName);

  void fixTypeLoc(const ast_matchers::MatchFinder::MatchResult &Result,
                  SourceLocation Start, SourceLocation End, TypeLoc Type);

  // Information about moving an old namespace.
  struct MoveNamespace {
    // The start offset of the namespace block being moved in the original
    // code.
    unsigned Offset;
    // The length of the namespace block in the original code.
    unsigned Length;
    // The offset at which the new namespace block will be inserted in the
    // original code.
    unsigned InsertionOffset;
    // The file in which the namespace is declared.
    FileID FID;
    SourceManager *SourceMgr;
  };

  // Information about inserting a class forward declaration.
  struct InsertForwardDeclaration {
    // The offset at while the forward declaration will be inserted in the
    // original code.
    unsigned InsertionOffset;
    // The code to be inserted.
    std::string ForwardDeclText;
  };

  std::string FallbackStyle;
  // In match callbacks, this contains replacements for replacing `typeLoc`s in
  // and deleting forward declarations in the moved namespace blocks.
  // In `onEndOfTranslationUnit` callback, the previous added replacements are
  // applied (on the moved namespace blocks), and then changed code in old
  // namespaces re moved to new namespaces, and previously deleted forward
  // declarations are inserted back to old namespaces, from which they are
  // deleted.
  std::map<std::string, tooling::Replacements> &FileToReplacements;
  // A fully qualified name of the old namespace without "::" prefix, e.g.
  // "a::b::c".
  std::string OldNamespace;
  // A fully qualified name of the new namespace without "::" prefix, e.g.
  // "x::y::z".
  std::string NewNamespace;
  // The longest suffix in the old namespace that does not overlap the new
  // namespace.
  // For example, if `OldNamespace` is "a::b::c" and `NewNamespace` is
  // "a::x::y", then `DiffOldNamespace` will be "b::c".
  std::string DiffOldNamespace;
  // The longest suffix in the new namespace that does not overlap the old
  // namespace.
  // For example, if `OldNamespace` is "a::b::c" and `NewNamespace` is
  // "a::x::y", then `DiffNewNamespace` will be "x::y".
  std::string DiffNewNamespace;
  // A regex pattern that matches files to be processed.
  std::string FilePattern;
  // Information about moved namespaces grouped by file.
  // Since we are modifying code in old namespaces (e.g. add namespace
  // spedifiers) as well as moving them, we store information about namespaces
  // to be moved and only move them after all modifications are finished (i.e.
  // in `onEndOfTranslationUnit`).
  std::map<std::string, std::vector<MoveNamespace>> MoveNamespaces;
  // Information about forward declaration insertions grouped by files.
  // A class forward declaration is not moved, so it will be deleted from the
  // moved code block and inserted back into the old namespace. The insertion
  // will be done after removing the code from the old namespace and before
  // inserting it to the new namespace.
  std::map<std::string, std::vector<InsertForwardDeclaration>> InsertFwdDecls;
};

} // namespace change_namespace
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CHANGE_NAMESPACE_CHANGENAMESPACE_H
