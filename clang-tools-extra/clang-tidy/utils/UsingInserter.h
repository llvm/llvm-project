//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_USINGINSERTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_USINGINSERTER_H

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include <optional>
#include <set>

namespace clang::tidy::utils {

// UsingInserter adds using declarations for |QualifiedName| to the surrounding
// function.
// This allows using a shorter name without clobbering other scopes.
class UsingInserter {
public:
  UsingInserter(const SourceManager &SourceMgr);

  // Creates a \p using declaration fixit. Returns ``std::nullopt`` on error
  // or if the using declaration already exists.
  std::optional<FixItHint>
  createUsingDeclaration(ASTContext &Context, const Stmt &Statement,
                         llvm::StringRef QualifiedName);

  // Returns the unqualified version of the name if there is an
  // appropriate using declaration and the qualified name otherwise.
  llvm::StringRef getShortName(ASTContext &Context, const Stmt &Statement,
                               llvm::StringRef QualifiedName);

private:
  using NameInFunction = std::pair<const FunctionDecl *, std::string>;
  const SourceManager &SourceMgr;
  std::set<NameInFunction> AddedUsing;
};

} // namespace clang::tidy::utils
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_USINGINSERTER_H
