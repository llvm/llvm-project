//===----------------------- FunctionPointer.cpp ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionPointer.h"

namespace clang {
namespace interp {

APValue FunctionPointer::toAPValue(const ASTContext &) const {
  if (!Func)
    return APValue(static_cast<Expr *>(nullptr), CharUnits::Zero(), {},
                   /*OnePastTheEnd=*/false, /*IsNull=*/true);

  if (Func->getDecl())
    return APValue(Func->getDecl(), CharUnits::fromQuantity(0), {},
                   /*OnePastTheEnd=*/false, /*IsNull=*/false);
  return APValue(Func->getExpr(), CharUnits::fromQuantity(0), {},
                 /*OnePastTheEnd=*/false, /*IsNull=*/false);
}

void FunctionPointer::print(llvm::raw_ostream &OS) const {
  OS << "FnPtr(";
  if (Func)
    OS << Func->getName();
  else if (Func)
    OS << reinterpret_cast<uintptr_t>(Func);
  else
    OS << "nullptr";
  OS << ")";
}

} // namespace interp
} // namespace clang
