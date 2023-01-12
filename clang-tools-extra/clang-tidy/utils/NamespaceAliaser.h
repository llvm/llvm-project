//===---------- NamespaceAliaser.h - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_NAMESPACEALIASER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_NAMESPACEALIASER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <map>

namespace clang {
namespace tidy {
namespace utils {

// This class creates function-level namespace aliases.
class NamespaceAliaser {
public:
  explicit NamespaceAliaser(const SourceManager &SourceMgr);
  // Adds a namespace alias for \p Namespace valid near \p
  // Statement. Picks the first available name from \p Abbreviations.
  // Returns ``std::nullopt`` if an alias already exists or there is an error.
  llvm::Optional<FixItHint>
  createAlias(ASTContext &Context, const Stmt &Statement,
              llvm::StringRef Namespace,
              const std::vector<std::string> &Abbreviations);

  // Get an alias name for \p Namespace valid at \p Statement. Returns \p
  // Namespace if there is no alias.
  std::string getNamespaceName(ASTContext &Context, const Stmt &Statement,
                               llvm::StringRef Namespace) const;

private:
  const SourceManager &SourceMgr;
  llvm::DenseMap<const FunctionDecl *, llvm::StringMap<std::string>>
      AddedAliases;
};

} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_NAMESPACEALIASER_H
