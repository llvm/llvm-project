//===-- lib/Semantics/check-stop.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_STOP_H_
#define FORTRAN_SEMANTICS_CHECK_STOP_H_

#include "flang/Semantics/semantics.h"

namespace Fortran::parser {
struct StopStmt;
}

namespace Fortran::semantics {

// Semantic analysis of STOP and ERROR STOP statements.
class StopChecker : public virtual BaseChecker {
public:
  explicit StopChecker(SemanticsContext &context) : context_{context} {};

  void Enter(const parser::StopStmt &);

private:
  SemanticsContext &context_;
};

} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_CHECK_STOP_H_
