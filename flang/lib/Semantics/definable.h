//===-- lib/Semantics/definable.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_DEFINABLE_H_
#define FORTRAN_SEMANTICS_DEFINABLE_H_

// Utilities for checking the definability of variables and pointers in context,
// including checks for attempted definitions in PURE subprograms.
// Fortran 2018 C1101, C1158, C1594, &c.

#include "flang/Common/enum-set.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/expression.h"
#include "flang/Parser/char-block.h"
#include "flang/Parser/message.h"
#include <optional>

namespace Fortran::semantics {

class Symbol;
class Scope;

ENUM_CLASS(DefinabilityFlag,
    VectorSubscriptIsOk, // a vector subscript may appear (i.e., assignment)
    DuplicatesAreOk, // vector subscript may have duplicates
    PointerDefinition, // a pointer is being defined, not its target
    AcceptAllocatable, // treat allocatable as if it were a pointer
    PolymorphicOkInPure) // don't check for polymorphic type in pure subprogram

using DefinabilityFlags =
    common::EnumSet<DefinabilityFlag, DefinabilityFlag_enumSize>;

// Tests a symbol or LHS variable or pointer for definability in a given scope.
// When the entity is not definable, returns a "because:" Message suitable for
// attachment to an error message to explain why the entity cannot be defined.
// When the entity can be defined in that context, returns std::nullopt.
std::optional<parser::Message> WhyNotDefinable(
    parser::CharBlock, const Scope &, DefinabilityFlags, const Symbol &);
std::optional<parser::Message> WhyNotDefinable(parser::CharBlock, const Scope &,
    DefinabilityFlags, const evaluate::Expr<evaluate::SomeType> &);

// If a symbol would not be definable in a pure scope, or not be usable as the
// target of a pointer assignment in a pure scope, return a constant string
// describing why.
const char *WhyBaseObjectIsSuspicious(const Symbol &, const Scope &);

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_DEFINABLE_H_
