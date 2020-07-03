//===--- QuerySession.h - clang-query ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_SESSION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_SESSION_H

#include "clang/AST/ASTTypeTraits.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"

namespace clang {

class ASTUnit;

namespace query {

/// Represents the state for a particular clang-query session.
class QuerySession {
public:
  QuerySession(llvm::ArrayRef<std::unique_ptr<ASTUnit>> ASTs)
      : ASTs(ASTs), PrintOutput(false), DiagOutput(true),
        DetailedASTOutput(false), BindRoot(true), PrintMatcher(false),
        Terminate(false), TK(ast_type_traits::TK_AsIs) {}

  llvm::ArrayRef<std::unique_ptr<ASTUnit>> ASTs;

  bool PrintOutput;
  bool DiagOutput;
  bool DetailedASTOutput;

  bool BindRoot;
  bool PrintMatcher;
  bool Terminate;

  ast_type_traits::TraversalKind TK;
  llvm::StringMap<ast_matchers::dynamic::VariantValue> NamedValues;
};

} // namespace query
} // namespace clang

#endif
