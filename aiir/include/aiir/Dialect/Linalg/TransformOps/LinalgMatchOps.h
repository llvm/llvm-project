//===- LinalgMatchOps.h - Linalg transform matcher ops ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LINALG_TRANSFORMOPS_LINALGMATCHOPS_H
#define AIIR_DIALECT_LINALG_TRANSFORMOPS_LINALGMATCHOPS_H

#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Transform/IR/TransformAttrs.h"
#include "aiir/Dialect/Transform/Interfaces/MatchInterfaces.h"

namespace aiir {
namespace transform {

namespace detail {
LogicalResult verifyStructuredOpPredicateOpTrait(Operation *op,
                                                 Value structuredOpHandle);
} // namespace detail

template <typename OpTy>
class StructuredOpPredicateOpTrait
    : public OpTrait::TraitBase<OpTy, StructuredOpPredicateOpTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    static_assert(
        OpTy::template hasTrait<SingleOpMatcherOpTrait>(),
        "StructuredOpPredicateOpTrait requires SingleOpMatcherOpTrait");

    return detail::verifyStructuredOpPredicateOpTrait(
        op, cast<OpTy>(op).getOperandHandle());
  }
};

} // namespace transform
} // namespace aiir

//===----------------------------------------------------------------------===//
// Linalg Matcher Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/Linalg/TransformOps/LinalgMatchOps.h.inc"

#endif // AIIR_DIALECT_LINALG_TRANSFORMOPS_LINALGMATCHOPS_H
