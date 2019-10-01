//===- ExpressionTraits.h - C++ Expression Traits Support Enums -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines enumerations for expression traits intrinsics.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_EXPRESSIONTRAITS_H
#define LLVM_CLANG_BASIC_EXPRESSIONTRAITS_H

namespace clang {

  enum ExpressionTrait {
    ET_IsLValueExpr,
    ET_IsRValueExpr
  };
}

#endif
