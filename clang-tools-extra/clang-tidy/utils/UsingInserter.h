//===---------- UsingInserter.h - clang-tidy ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_USINGINSERTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_USINGINSERTER_H

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include <set>

namespace clang {
namespace tidy {
namespace utils {

// UsingInserter adds using declarations for |QualifiedName| to the surrounding
// function.
// This allows using a shorter name without clobbering other scopes.
class UsingInserter {
public:
  UsingInserter(const SourceManager &SourceMgr);

  // Creates a \p using declaration fixit. Returns ``llvm::None`` on error
  // or if the using declaration already exists.
  llvm::Optional<FixItHint>
  createUsingDeclaration(ASTContext &Context, const Stmt &Statement,
                         llvm::StringRef QualifiedName);

  // Returns the unqualified version of the name if there is an
  // appropriate using declaration and the qualified name otherwise.
  llvm::StringRef getShortName(ASTContext &Context, const Stmt &Statement,
                               llvm::StringRef QualifiedName);

private:
  typedef std::pair<const FunctionDecl *, std::string> NameInFunction;
  const SourceManager &SourceMgr;
  std::set<NameInFunction> AddedUsing;
};

} // namespace utils
} // namespace tidy
} // namespace clang
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_USINGINSERTER_H
