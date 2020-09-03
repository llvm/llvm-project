//===-- lib/Semantics/check-select-type.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_SELECT_TYPE_H_
#define FORTRAN_SEMANTICS_CHECK_SELECT_TYPE_H_

#include "flang/Semantics/semantics.h"

namespace Fortran::parser {
struct SelectTypeConstruct;
struct Selector;
} // namespace Fortran::parser

namespace Fortran::semantics {

class SelectTypeChecker : public virtual BaseChecker {
public:
  explicit SelectTypeChecker(SemanticsContext &context) : context_{context} {};
  void Enter(const parser::SelectTypeConstruct &);

private:
  const SomeExpr *GetExprFromSelector(const parser::Selector &);
  SemanticsContext &context_;
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_SELECT_TYPE_H_
