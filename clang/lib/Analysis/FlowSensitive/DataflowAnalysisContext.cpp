//===-- DataflowAnalysisContext.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a DataflowAnalysisContext class that owns objects that
//  encompass the state of a program and stores context that is used during
//  dataflow analysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include <cassert>
#include <memory>
#include <utility>

namespace clang {
namespace dataflow {

static std::pair<BoolValue *, BoolValue *>
makeCanonicalBoolValuePair(BoolValue &LHS, BoolValue &RHS) {
  auto Res = std::make_pair(&LHS, &RHS);
  if (&RHS < &LHS)
    std::swap(Res.first, Res.second);
  return Res;
}

BoolValue &
DataflowAnalysisContext::getOrCreateConjunctionValue(BoolValue &LHS,
                                                     BoolValue &RHS) {
  if (&LHS == &RHS)
    return LHS;

  auto Res = ConjunctionVals.try_emplace(makeCanonicalBoolValuePair(LHS, RHS),
                                         nullptr);
  if (Res.second)
    Res.first->second =
        &takeOwnership(std::make_unique<ConjunctionValue>(LHS, RHS));
  return *Res.first->second;
}

BoolValue &
DataflowAnalysisContext::getOrCreateDisjunctionValue(BoolValue &LHS,
                                                     BoolValue &RHS) {
  if (&LHS == &RHS)
    return LHS;

  auto Res = DisjunctionVals.try_emplace(makeCanonicalBoolValuePair(LHS, RHS),
                                         nullptr);
  if (Res.second)
    Res.first->second =
        &takeOwnership(std::make_unique<DisjunctionValue>(LHS, RHS));
  return *Res.first->second;
}

BoolValue &DataflowAnalysisContext::getOrCreateNegationValue(BoolValue &Val) {
  auto Res = NegationVals.try_emplace(&Val, nullptr);
  if (Res.second)
    Res.first->second = &takeOwnership(std::make_unique<NegationValue>(Val));
  return *Res.first->second;
}

} // namespace dataflow
} // namespace clang
