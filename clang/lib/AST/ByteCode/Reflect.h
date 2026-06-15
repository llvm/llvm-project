//===--- Boolean.h - Wrapper for boolean types for the VM -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_REFLECT_H
#define LLVM_CLANG_AST_INTERP_REFLECT_H

#include "clang/AST/APValue.h"
#include "clang/AST/ComparisonCategories.h"
#include "clang/AST/Reflection.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>

namespace clang {
namespace interp {

class Reflect final {
private:
  ReflectionKind Kind;
  const void *Operand;

public:
  Reflect() : Kind(ReflectionKind::Null), Operand(nullptr) {}
  Reflect(ReflectionKind Kind, const void *Operand)
      : Kind(Kind), Operand(Operand) {}

  ComparisonCategoryResult compare(const Reflect &RHS) const {
    llvm::FoldingSetNodeID LID, RID;
    APValue(Kind, Operand).Profile(LID);
    APValue(RHS.Kind, RHS.Operand).Profile(RID);

    if (LID == RID)
      return ComparisonCategoryResult::Equal;
    return ComparisonCategoryResult::Unordered;
  }

  void print(llvm::raw_ostream &OS) const {
    OS << "Reflect(" << Kind << ", " << Operand << ")";
  }
  APValue toAPValue(const ASTContext &Ctx) const {
    return APValue(Kind, Operand);
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Reflect &R) {
  R.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
