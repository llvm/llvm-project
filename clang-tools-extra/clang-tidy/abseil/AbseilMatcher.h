//===- AbseilMatcher.h - clang-tidy ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace ast_matchers {

/// Matches AST nodes that were found within Abseil files.
///
/// Example matches Y but not X
///     (matcher = cxxRecordDecl(isInAbseilFile())
/// \code
///   #include "absl/strings/internal-file.h"
///   class X {};
/// \endcode
/// absl/strings/internal-file.h:
/// \code
///   class Y {};
/// \endcode
///
/// Usable as: Matcher<Decl>, Matcher<Stmt>, Matcher<TypeLoc>,
/// Matcher<NestedNameSpecifierLoc>

AST_POLYMORPHIC_MATCHER(isInAbseilFile,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(
                            Decl, Stmt, TypeLoc, NestedNameSpecifierLoc)) {
  auto &SourceManager = Finder->getASTContext().getSourceManager();
  SourceLocation Loc = Node.getBeginLoc();
  if (Loc.isInvalid())
    return false;
  const FileEntry *FileEntry =
      SourceManager.getFileEntryForID(SourceManager.getFileID(Loc));
  if (!FileEntry)
    return false;
  StringRef Filename = FileEntry->getName();
  llvm::Regex RE(
      "absl/(algorithm|base|container|debugging|memory|meta|numeric|strings|"
      "synchronization|time|types|utility)");
  return RE.match(Filename);
}

} // namespace ast_matchers
} // namespace clang
