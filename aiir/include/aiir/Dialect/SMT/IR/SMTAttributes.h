//===- SMTAttributes.h - Declare SMT dialect attributes ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SMT_IR_SMTATTRIBUTES_H
#define AIIR_DIALECT_SMT_IR_SMTATTRIBUTES_H

#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributeInterfaces.h"
#include "aiir/IR/BuiltinAttributes.h"

namespace aiir {
namespace smt {
namespace detail {

struct BitVectorAttrStorage;

} // namespace detail
} // namespace smt
} // namespace aiir

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/SMT/IR/SMTAttributes.h.inc"

#endif // AIIR_DIALECT_SMT_IR_SMTATTRIBUTES_H
