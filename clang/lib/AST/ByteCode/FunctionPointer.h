//===--- FunctionPointer.h - Types for the constexpr VM ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_FUNCTION_POINTER_H
#define LLVM_CLANG_AST_INTERP_FUNCTION_POINTER_H

#include "Function.h"
#include "Primitives.h"

namespace clang {
class ASTContext;
class APValue;
namespace interp {

class FunctionPointer final {
private:
  const Function *Func;
  uint64_t Offset;
  bool Valid;

public:
  FunctionPointer() = default;
  FunctionPointer(const Function *Func, uint64_t Offset = 0)
      : Func(Func), Offset(Offset), Valid(true) {}

  FunctionPointer(uintptr_t IntVal, const Descriptor *Desc = nullptr)
      : Func(reinterpret_cast<const Function *>(IntVal)), Offset(0),
        Valid(false) {}

  const Function *getFunction() const { return Func; }
  uint64_t getOffset() const { return Offset; }
  bool isZero() const { return !Func; }
  bool isValid() const { return Valid; }
  bool isWeak() const {
    if (!Func || !Valid || !Func->getDecl())
      return false;

    return Func->getDecl()->isWeak();
  }

  APValue toAPValue(const ASTContext &) const;
  void print(llvm::raw_ostream &OS) const;

  std::string toDiagnosticString(const ASTContext &Ctx) const {
    if (!Func)
      return "nullptr";

    return toAPValue(Ctx).getAsString(Ctx, Func->getDecl()->getType());
  }

  uint64_t getIntegerRepresentation() const {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Func));
  }

  ComparisonCategoryResult compare(const FunctionPointer &RHS) const {
    if (Func == RHS.Func && Offset == RHS.Offset)
      return ComparisonCategoryResult::Equal;
    return ComparisonCategoryResult::Unordered;
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     FunctionPointer FP) {
  FP.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
