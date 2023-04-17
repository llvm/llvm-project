//===-- Arena.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Arena.h"

namespace clang::dataflow {

static std::pair<BoolValue *, BoolValue *>
makeCanonicalBoolValuePair(BoolValue &LHS, BoolValue &RHS) {
  auto Res = std::make_pair(&LHS, &RHS);
  if (&RHS < &LHS)
    std::swap(Res.first, Res.second);
  return Res;
}

BoolValue &Arena::makeAnd(BoolValue &LHS, BoolValue &RHS) {
  if (&LHS == &RHS)
    return LHS;

  auto Res = ConjunctionVals.try_emplace(makeCanonicalBoolValuePair(LHS, RHS),
                                         nullptr);
  if (Res.second)
    Res.first->second = &create<ConjunctionValue>(LHS, RHS);
  return *Res.first->second;
}

BoolValue &Arena::makeOr(BoolValue &LHS, BoolValue &RHS) {
  if (&LHS == &RHS)
    return LHS;

  auto Res = DisjunctionVals.try_emplace(makeCanonicalBoolValuePair(LHS, RHS),
                                         nullptr);
  if (Res.second)
    Res.first->second = &create<DisjunctionValue>(LHS, RHS);
  return *Res.first->second;
}

BoolValue &Arena::makeNot(BoolValue &Val) {
  auto Res = NegationVals.try_emplace(&Val, nullptr);
  if (Res.second)
    Res.first->second = &create<NegationValue>(Val);
  return *Res.first->second;
}

BoolValue &Arena::makeImplies(BoolValue &LHS, BoolValue &RHS) {
  if (&LHS == &RHS)
    return makeLiteral(true);

  auto Res = ImplicationVals.try_emplace(std::make_pair(&LHS, &RHS), nullptr);
  if (Res.second)
    Res.first->second = &create<ImplicationValue>(LHS, RHS);
  return *Res.first->second;
}

BoolValue &Arena::makeEquals(BoolValue &LHS, BoolValue &RHS) {
  if (&LHS == &RHS)
    return makeLiteral(true);

  auto Res = BiconditionalVals.try_emplace(makeCanonicalBoolValuePair(LHS, RHS),
                                           nullptr);
  if (Res.second)
    Res.first->second = &create<BiconditionalValue>(LHS, RHS);
  return *Res.first->second;
}

} // namespace clang::dataflow
