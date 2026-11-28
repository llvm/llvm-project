//===-- ValueTransferer.h - Transfer functions for nullability --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file contains transfer functions for flow-sensitive pointer nullability
// analysis. We call this "value transferer" to distinguish from "type
// transferer" (see
// https://github.com/google/crubit/blob/7ef99d77672f716792140a5d5f6548cf7d638200/nullability/type_transferer.h#L13)
// In contrast, the type transferer only handles flow-insensitive properties
// derived from the types.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_VALUETRANSFERER_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_VALUETRANSFERER_H

#include "clang/AST/Expr.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/Models/Nullability/PointerNullabilityAnalysis.h"
#include "clang/Analysis/FlowSensitive/Value.h"

namespace clang::dataflow::nullability {

// Returns a switch over CFG elements that can be used to transfer
// (flow-sensitive) value properties, i.e. to update the analysis state to
// account for the impact of the CFG element.
CFGMatchSwitch<TransferState<PointerNullabilityLattice>> buildValueTransferer();

// If `E` is already associated with a `PointerValue`, returns it.
// Otherwise, associates a newly created `PointerValue` with `E` and returns it.
// Returns null iff `E` is not a raw pointer expression.
PointerValue *ensureRawPointerHasValue(const Expr *E, Environment &Env);

// Initialize the null state of `PointerVal` based on the nullability of the
// type of `E`.
void initPointerFromTypeNullability(
    PointerValue &PointerVal, const Expr *E,
    TransferState<PointerNullabilityLattice> &State);

} // namespace clang::dataflow::nullability

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_VALUETRANSFERER_H
