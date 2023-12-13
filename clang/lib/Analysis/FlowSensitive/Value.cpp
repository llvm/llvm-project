//===-- Value.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines support functions for the `Value` type.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/DebugSupport.h"
#include "llvm/Support/Casting.h"

namespace clang {
namespace dataflow {

static bool areEquivalentIndirectionValues(const Value &Val1,
                                           const Value &Val2) {
  if (auto *IndVal1 = dyn_cast<PointerValue>(&Val1)) {
    auto *IndVal2 = cast<PointerValue>(&Val2);
    return &IndVal1->getPointeeLoc() == &IndVal2->getPointeeLoc();
  }
  return false;
}

bool areEquivalentValues(const Value &Val1, const Value &Val2) {
  return &Val1 == &Val2 || (Val1.getKind() == Val2.getKind() &&
                            (isa<TopBoolValue>(&Val1) ||
                             areEquivalentIndirectionValues(Val1, Val2)));
}

raw_ostream &operator<<(raw_ostream &OS, const Value &Val) {
  switch (Val.getKind()) {
  case Value::Kind::Integer:
    return OS << "Integer(@" << &Val << ")";
  case Value::Kind::Pointer:
    return OS << "Pointer(" << &cast<PointerValue>(Val).getPointeeLoc() << ")";
  case Value::Kind::Record:
    return OS << "Record(" << &cast<RecordValue>(Val).getLoc() << ")";
  case Value::Kind::TopBool:
    return OS << "TopBool(" << cast<TopBoolValue>(Val).getAtom() << ")";
  case Value::Kind::AtomicBool:
    return OS << "AtomicBool(" << cast<AtomicBoolValue>(Val).getAtom() << ")";
  case Value::Kind::FormulaBool:
    return OS << "FormulaBool(" << cast<FormulaBoolValue>(Val).formula() << ")";
  }
  llvm_unreachable("Unknown clang::dataflow::Value::Kind enum");
}

} // namespace dataflow
} // namespace clang
