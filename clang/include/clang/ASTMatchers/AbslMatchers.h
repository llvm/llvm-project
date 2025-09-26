//===- AbslMatchers.h - AST Matchers for Abseil -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements matchers specific to structures in Abseil
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ASTMATCHERS_ABSLMATCHERS_H
#define LLVM_CLANG_ASTMATCHERS_ABSLMATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang {
namespace ast_matchers {
namespace absl_matchers {

DeclarationMatcher statusOrClass();
DeclarationMatcher statusClass();

} // end namespace absl_matchers
} // end namespace ast_matchers
} // end namespace clang

#endif // LLVM_CLANG_ASTMATCHERS_ABSLMATCHERS_H
