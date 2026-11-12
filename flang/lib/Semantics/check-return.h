//===-- lib/Semantics/check-return.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_RETURN_H_
#define FORTRAN_SEMANTICS_CHECK_RETURN_H_

#include "flang/Semantics/semantics.h"

namespace Fortran::parser {
struct ReturnStmt;
}

namespace Fortran::semantics {
class ReturnStmtChecker : public virtual BaseChecker {
public:
  ReturnStmtChecker(SemanticsContext &context) : context_{context} {}
  void Leave(const parser::ReturnStmt &);

private:
  SemanticsContext &context_;
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_RETURN_H_
