//===--- QuerySession.h - clang-query ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_SESSION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_QUERY_QUERY_SESSION_H

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
        Terminate(false) {}

  llvm::ArrayRef<std::unique_ptr<ASTUnit>> ASTs;

  bool PrintOutput;
  bool DiagOutput;
  bool DetailedASTOutput;

  bool BindRoot;
  bool PrintMatcher;
  bool Terminate;
  llvm::StringMap<ast_matchers::dynamic::VariantValue> NamedValues;
};

} // namespace query
} // namespace clang

#endif
