//===-- lib/Parser/expr-parsers.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_EXPR_PARSERS_H_
#define FORTRAN_PARSER_EXPR_PARSERS_H_

#include "basic-parsers.h"
#include "token-parsers.h"
#include "type-parsers.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::parser {

// R403 scalar-xyz -> xyz
// Also define constant-xyz, int-xyz, default-char-xyz.
template <typename PA> inline constexpr auto scalar(const PA &p) {
  return construct<Scalar<typename PA::resultType>>(p); // scalar-p
}

template <typename PA> inline constexpr auto constant(const PA &p) {
  return construct<Constant<typename PA::resultType>>(p); // constant-p
}

template <typename PA> inline constexpr auto integer(const PA &p) {
  return construct<Integer<typename PA::resultType>>(p); // int-p
}

template <typename PA> inline constexpr auto logical(const PA &p) {
  return construct<Logical<typename PA::resultType>>(p); // logical-p
}

template <typename PA> inline constexpr auto defaultChar(const PA &p) {
  return construct<DefaultChar<typename PA::resultType>>(p); // default-char-p
}

// N.B. charLiteralConstantWithoutKind does not skip preceding space.
constexpr auto charLiteralConstantWithoutKind{
    "'"_ch >> CharLiteral<'\''>{} || "\""_ch >> CharLiteral<'"'>{}};

// R904 logical-variable -> variable
// Appears only as part of scalar-logical-variable.
constexpr auto scalarLogicalVariable{scalar(logical(variable))};

// R906 default-char-variable -> variable
// Appears only as part of scalar-default-char-variable.
constexpr auto scalarDefaultCharVariable{scalar(defaultChar(variable))};

// R907 int-variable -> variable
// Appears only as part of scalar-int-variable.
constexpr auto scalarIntVariable{scalar(integer(variable))};

// R930 errmsg-variable -> scalar-default-char-variable
// R1207 iomsg-variable -> scalar-default-char-variable
constexpr auto msgVariable{construct<MsgVariable>(scalarDefaultCharVariable)};

// R1024 logical-expr -> expr
constexpr auto logicalExpr{logical(indirect(expr))};
constexpr auto scalarLogicalExpr{scalar(logicalExpr)};

// R1025 default-char-expr -> expr
constexpr auto defaultCharExpr{defaultChar(indirect(expr))};
constexpr auto scalarDefaultCharExpr{scalar(defaultCharExpr)};

// R1026 int-expr -> expr
constexpr auto intExpr{integer(indirect(expr))};
constexpr auto scalarIntExpr{scalar(intExpr)};

// R1029 constant-expr -> expr
constexpr auto constantExpr{constant(indirect(expr))};
constexpr auto scalarExpr{scalar(indirect(expr))};

// R1030 default-char-constant-expr -> default-char-expr
constexpr auto scalarDefaultCharConstantExpr{scalar(defaultChar(constantExpr))};

// R1031 int-constant-expr -> int-expr
constexpr auto intConstantExpr{integer(constantExpr)};
constexpr auto scalarIntConstantExpr{scalar(intConstantExpr)};

// R935 lower-bound-expr -> scalar-int-expr
// R936 upper-bound-expr -> scalar-int-expr
constexpr auto boundExpr{scalarIntExpr};

// R1115 team-value -> scalar-expr
constexpr auto teamValue{scalar(indirect(expr))};

// R1124 do-variable -> scalar-int-variable-name
constexpr auto doVariable{scalar(integer(name))};

// NOTE: In loop-control we allow REAL name and bounds too.
// This means parse them without the integer constraint and check later.
inline constexpr auto loopBounds(decltype(scalarExpr) &p) {
  return construct<LoopBounds<ScalarName, ScalarExpr>>(
      scalar(name) / "=", p / ",", p, maybe("," >> p));
}
template <typename PA> inline constexpr auto loopBounds(const PA &p) {
  return construct<LoopBounds<DoVariable, typename PA::resultType>>(
      doVariable / "=", p / ",", p, maybe("," >> p));
}
} // namespace Fortran::parser
#endif
