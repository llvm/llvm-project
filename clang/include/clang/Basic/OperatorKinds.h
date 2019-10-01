//===--- OperatorKinds.h - C++ Overloaded Operators -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines an enumeration for C++ overloaded operators.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPERATORKINDS_H
#define LLVM_CLANG_BASIC_OPERATORKINDS_H

namespace clang {

/// Enumeration specifying the different kinds of C++ overloaded
/// operators.
enum OverloadedOperatorKind : int {
  OO_None,                ///< Not an overloaded operator
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
  OO_##Name,
#include "clang/Basic/OperatorKinds.def"
  NUM_OVERLOADED_OPERATORS
};

/// Retrieve the spelling of the given overloaded operator, without
/// the preceding "operator" keyword.
const char *getOperatorSpelling(OverloadedOperatorKind Operator);

} // end namespace clang

#endif
