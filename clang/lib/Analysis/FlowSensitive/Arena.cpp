//===-- Arena.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/Value.h"

namespace clang::dataflow {

static std::pair<const Formula *, const Formula *>
canonicalFormulaPair(const Formula &LHS, const Formula &RHS) {
  auto Res = std::make_pair(&LHS, &RHS);
  if (&RHS < &LHS) // FIXME: use a deterministic order instead
    std::swap(Res.first, Res.second);
  return Res;
}

const Formula &Arena::makeAtomRef(Atom A) {
  auto [It, Inserted] = AtomRefs.try_emplace(A);
  if (Inserted)
    It->second =
        &Formula::create(Alloc, Formula::AtomRef, {}, static_cast<unsigned>(A));
  return *It->second;
}

const Formula &Arena::makeAnd(const Formula &LHS, const Formula &RHS) {
  if (&LHS == &RHS)
    return LHS;

  auto [It, Inserted] =
      Ands.try_emplace(canonicalFormulaPair(LHS, RHS), nullptr);
  if (Inserted)
    It->second = &Formula::create(Alloc, Formula::And, {&LHS, &RHS});
  return *It->second;
}

const Formula &Arena::makeOr(const Formula &LHS, const Formula &RHS) {
  if (&LHS == &RHS)
    return LHS;

  auto [It, Inserted] =
      Ors.try_emplace(canonicalFormulaPair(LHS, RHS), nullptr);
  if (Inserted)
    It->second = &Formula::create(Alloc, Formula::Or, {&LHS, &RHS});
  return *It->second;
}

const Formula &Arena::makeNot(const Formula &Val) {
  auto [It, Inserted] = Nots.try_emplace(&Val, nullptr);
  if (Inserted)
    It->second = &Formula::create(Alloc, Formula::Not, {&Val});
  return *It->second;
}

const Formula &Arena::makeImplies(const Formula &LHS, const Formula &RHS) {
  if (&LHS == &RHS)
    return makeLiteral(true);

  auto [It, Inserted] =
      Implies.try_emplace(std::make_pair(&LHS, &RHS), nullptr);
  if (Inserted)
    It->second = &Formula::create(Alloc, Formula::Implies, {&LHS, &RHS});
  return *It->second;
}

const Formula &Arena::makeEquals(const Formula &LHS, const Formula &RHS) {
  if (&LHS == &RHS)
    return makeLiteral(true);

  auto [It, Inserted] =
      Equals.try_emplace(canonicalFormulaPair(LHS, RHS), nullptr);
  if (Inserted)
    It->second = &Formula::create(Alloc, Formula::Equal, {&LHS, &RHS});
  return *It->second;
}

IntegerValue &Arena::makeIntLiteral(llvm::APInt Value) {
  auto [It, Inserted] = IntegerLiterals.try_emplace(Value, nullptr);

  if (Inserted)
    It->second = &create<IntegerValue>();
  return *It->second;
}

BoolValue &Arena::makeBoolValue(const Formula &F) {
  auto [It, Inserted] = FormulaValues.try_emplace(&F);
  if (Inserted)
    It->second = (F.kind() == Formula::AtomRef)
                     ? (BoolValue *)&create<AtomicBoolValue>(F)
                     : &create<FormulaBoolValue>(F);
  return *It->second;
}

} // namespace clang::dataflow
