//===-- lib/Semantics/check-case.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_CASE_H_
#define FORTRAN_SEMANTICS_CHECK_CASE_H_

#include "flang/Semantics/semantics.h"

namespace Fortran::parser {
struct CaseConstruct;
}

namespace Fortran::semantics {

class CaseChecker : public virtual BaseChecker {
public:
  explicit CaseChecker(SemanticsContext &context) : context_{context} {};

  void Enter(const parser::CaseConstruct &);

private:
  SemanticsContext &context_;
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_CASE_H_
