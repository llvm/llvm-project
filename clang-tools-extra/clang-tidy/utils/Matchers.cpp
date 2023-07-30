//===---------- Matchers.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Matchers.h"
#include "ASTUtils.h"

namespace clang::tidy::matchers {

bool NotIdenticalStatementsPredicate::operator()(
    const clang::ast_matchers::internal::BoundNodesMap &Nodes) const {
  return !utils::areStatementsIdentical(Node.get<Stmt>(),
                                        Nodes.getNodeAs<Stmt>(ID), *Context);
}

} // namespace clang::tidy::matchers
