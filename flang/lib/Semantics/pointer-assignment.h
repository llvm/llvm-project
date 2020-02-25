//===-- lib/Semantics/pointer-assignment.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_POINTER_ASSIGNMENT_H_
#define FORTRAN_SEMANTICS_POINTER_ASSIGNMENT_H_

#include "flang/Evaluate/expression.h"
#include "flang/Parser/char-block.h"
#include "flang/Semantics/type.h"
#include <string>

namespace Fortran::evaluate::characteristics {
struct DummyDataObject;
}

namespace Fortran::evaluate {
class FoldingContext;
}

namespace Fortran::semantics {

class Symbol;

void CheckPointerAssignment(
    evaluate::FoldingContext &, const evaluate::Assignment &);
void CheckPointerAssignment(
    evaluate::FoldingContext &, const Symbol &lhs, const SomeExpr &rhs);
void CheckPointerAssignment(evaluate::FoldingContext &,
    parser::CharBlock source, const std::string &description,
    const evaluate::characteristics::DummyDataObject &, const SomeExpr &rhs);

}

#endif  // FORTRAN_SEMANTICS_POINTER_ASSIGNMENT_H_
