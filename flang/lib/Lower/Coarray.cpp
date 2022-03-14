//===-- Coarray.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implementation of the lowering of image related constructs and expressions.
/// Fortran images can form teams, communicate via coarrays, etc.
///
//===----------------------------------------------------------------------===//

#include "flang/Lower/Coarray.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"

//===----------------------------------------------------------------------===//
// TEAM statements and constructs
//===----------------------------------------------------------------------===//

void Fortran::lower::genChangeTeamConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::ChangeTeamConstruct &) {
  TODO(converter.getCurrentLocation(), "CHANGE TEAM construct");
}

void Fortran::lower::genChangeTeamStmt(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::ChangeTeamStmt &) {
  TODO(converter.getCurrentLocation(), "CHANGE TEAM stmt");
}

void Fortran::lower::genEndChangeTeamStmt(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &,
    const Fortran::parser::EndChangeTeamStmt &) {
  TODO(converter.getCurrentLocation(), "END CHANGE TEAM");
}

void Fortran::lower::genFormTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &, const Fortran::parser::FormTeamStmt &) {
  TODO(converter.getCurrentLocation(), "FORM TEAM");
}

//===----------------------------------------------------------------------===//
// COARRAY expressions
//===----------------------------------------------------------------------===//

fir::ExtendedValue Fortran::lower::CoarrayExprHelper::genAddr(
    const Fortran::evaluate::CoarrayRef &expr) {
  (void)symMap;
  TODO(converter.getCurrentLocation(), "co-array address");
}

fir::ExtendedValue Fortran::lower::CoarrayExprHelper::genValue(
    const Fortran::evaluate::CoarrayRef &expr) {
  TODO(converter.getCurrentLocation(), "co-array value");
}
