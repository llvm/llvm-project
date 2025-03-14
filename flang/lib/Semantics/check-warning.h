//===-- lib/Semantics/check-warning.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_WARNING_H_
#define FORTRAN_SEMANTICS_CHECK_WARNING_H_

#include "flang/Semantics/semantics.h"

namespace Fortran::parser {
struct FunctionStmt;
struct InterfaceBody;
struct SubroutineStmt;
struct EntityDecl;
} // namespace Fortran::parser

namespace Fortran::semantics {

// Perform semantic checks on DummyArg on Function and Subroutine
// TODO: Add checks for future warning options
class WarningChecker : public virtual BaseChecker {
public:
  explicit WarningChecker(SemanticsContext &context) : context_{context} {}
  void Enter(const parser::FunctionStmt &);
  void Enter(const parser::SubroutineStmt &);
  void Enter(const parser::EntityDecl &);

private:
  SemanticsContext &context_;
};
} // namespace Fortran::semantics
#endif
