//===-- ComponentPath.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ComponentPath.h"

static std::function<
    Fortran::lower::IterationSpace(const Fortran::lower::IterationSpace &)>
getIdentityFunc() {
  return [](const Fortran::lower::IterationSpace &s) { return s; };
}

static std::function<
    Fortran::lower::IterationSpace(const Fortran::lower::IterationSpace &)>
getNullaryFunc() {
  return [](const Fortran::lower::IterationSpace &s) {
    Fortran::lower::IterationSpace newIters(s);
    newIters.clearIndices();
    return newIters;
  };
}

void Fortran::lower::ComponentPath::clear() {
  reversePath.clear();
  substring = nullptr;
  applied = false;
  prefixComponents.clear();
  trips.clear();
  suffixComponents.clear();
  pc = getIdentityFunc();
}

bool Fortran::lower::isRankedArrayAccess(const Fortran::evaluate::ArrayRef &x) {
  for (const Fortran::evaluate::Subscript &sub : x.subscript()) {
    if (Fortran::common::visit(
            Fortran::common::visitors{
                [&](const Fortran::evaluate::Triplet &) { return true; },
                [&](const Fortran::evaluate::IndirectSubscriptIntegerExpr &e) {
                  return e.value().Rank() > 0;
                }},
            sub.u))
      return true;
  }
  return false;
}

void Fortran::lower::ComponentPath::resetPC() { pc = getIdentityFunc(); }

void Fortran::lower::ComponentPath::setPC(bool isImplicit) {
  pc = isImplicit ? getIdentityFunc() : getNullaryFunc();
  resetExtendCoorRef();
}

Fortran::lower::ComponentPath::ExtendRefFunc
Fortran::lower::ComponentPath::getExtendCoorRef() const {
  return hasExtendCoorRef() ? *extendCoorRef : [](mlir::Value v) { return v; };
}
