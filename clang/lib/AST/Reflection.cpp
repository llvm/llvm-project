//=== Reflection.cpp - Implementation of ReflectionValue and helpers -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Reflection.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeBase.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace clang;

ReflectionValue::ReflectionValue(const TypeSourceInfo *TSI)
    : Kind(ReflectionKind::Type), Operand(TSI) {
  assert(TSI && "type reflection must not be null");
}

ReflectionValue ReflectionValue::CreateUnsafe(ReflectionKind Kind,
                                              const void *Operand) {
  return ReflectionValue(
      Kind, OperandType::getFromOpaqueValue(const_cast<void *>(Operand)));
}

const TypeSourceInfo *ReflectionValue::castAsType() const {
  assert(getKind() == ReflectionKind::Type);
  return cast<const TypeSourceInfo *>(getOperand());
}

void ReflectionValue::print(raw_ostream &Out, const PrintingPolicy &Policy) {
  switch (getKind()) {
  case ReflectionKind::Null:
    Out << "std::meta::info{}";
    return;
  case ReflectionKind::Type:
    Out << "^^" << castAsType()->getType().stream(Policy);
    return;
  }
  llvm_unreachable("invalid reflection kind");
}
