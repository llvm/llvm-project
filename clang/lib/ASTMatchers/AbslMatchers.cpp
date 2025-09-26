//===- AbslMatchers.cpp - AST Matchers for Abseil --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/AbslMatchers.h"

namespace clang {
namespace ast_matchers {
namespace absl_matchers {
DeclarationMatcher statusOrClass() {
  return classTemplateSpecializationDecl(
      hasName("::absl::StatusOr"),
      hasTemplateArgument(0, refersToType(type().bind("StatusOrValueType"))));
}

DeclarationMatcher statusClass() {
  return cxxRecordDecl(hasName("::absl::Status"));
}

} // end namespace absl_matchers
} // end namespace ast_matchers
} // end namespace clang
