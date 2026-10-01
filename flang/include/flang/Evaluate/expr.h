//===-- include/flang/Evaluate/expr.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_EXPR_H_
#define FORTRAN_EVALUATE_EXPR_H_

#include "common.h"
#include "constant.h"
#include "expression.h"
#include "formatting.h"
#include "type.h"
#include "variable.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Common/template.h"
#include "flang/Parser/char-block.h"
#include "flang/Support/Fortran.h"
#include <algorithm>
#include <tuple>
#include <type_traits>
#include <variant>

namespace Fortran::evaluate::expr {

class ActualArgument {};

} // namespace Fortran::evaluate::expr
#endif // FORTRAN_EVALUATE_EXPRESSION_H_
