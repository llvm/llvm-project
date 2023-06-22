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

template <class Key, class ComputeFunc>
const Formula &cached(llvm::DenseMap<Key, const Formula *> &Cache, Key K,
                      ComputeFunc &&Compute) {
  auto [It, Inserted] = Cache.try_emplace(std::forward<Key>(K));
  if (Inserted)
    It->second = Compute();
  return *It->second;
}

const Formula &Arena::makeAtomRef(Atom A) {
  return cached(AtomRefs, A, [&] {
    return &Formula::create(Alloc, Formula::AtomRef, {},
                            static_cast<unsigned>(A));
  });
}

const Formula &Arena::makeAnd(const Formula &LHS, const Formula &RHS) {
  return cached(Ands, canonicalFormulaPair(LHS, RHS), [&] {
    if (&LHS == &RHS)
      return &LHS;
    if (LHS.kind() == Formula::Literal)
      return LHS.literal() ? &RHS : &LHS;
    if (RHS.kind() == Formula::Literal)
      return RHS.literal() ? &LHS : &RHS;

    return &Formula::create(Alloc, Formula::And, {&LHS, &RHS});
  });
}

const Formula &Arena::makeOr(const Formula &LHS, const Formula &RHS) {
  return cached(Ors, canonicalFormulaPair(LHS, RHS), [&] {
    if (&LHS == &RHS)
      return &LHS;
    if (LHS.kind() == Formula::Literal)
      return LHS.literal() ? &LHS : &RHS;
    if (RHS.kind() == Formula::Literal)
      return RHS.literal() ? &RHS : &LHS;

    return &Formula::create(Alloc, Formula::Or, {&LHS, &RHS});
  });
}

const Formula &Arena::makeNot(const Formula &Val) {
  return cached(Nots, &Val, [&] {
    if (Val.kind() == Formula::Not)
      return Val.operands()[0];
    if (Val.kind() == Formula::Literal)
      return &makeLiteral(!Val.literal());

    return &Formula::create(Alloc, Formula::Not, {&Val});
  });
}

const Formula &Arena::makeImplies(const Formula &LHS, const Formula &RHS) {
  return cached(Implies, std::make_pair(&LHS, &RHS), [&] {
    if (&LHS == &RHS)
      return &makeLiteral(true);
    if (LHS.kind() == Formula::Literal)
      return LHS.literal() ? &RHS : &makeLiteral(true);
    if (RHS.kind() == Formula::Literal)
      return RHS.literal() ? &RHS : &makeNot(LHS);

    return &Formula::create(Alloc, Formula::Implies, {&LHS, &RHS});
  });
}

const Formula &Arena::makeEquals(const Formula &LHS, const Formula &RHS) {
  return cached(Equals, canonicalFormulaPair(LHS, RHS), [&] {
    if (&LHS == &RHS)
      return &makeLiteral(true);
    if (LHS.kind() == Formula::Literal)
      return LHS.literal() ? &RHS : &makeNot(RHS);
    if (RHS.kind() == Formula::Literal)
      return RHS.literal() ? &LHS : &makeNot(LHS);

    return &Formula::create(Alloc, Formula::Equal, {&LHS, &RHS});
  });
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
