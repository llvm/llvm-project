//===-- OpenMP.cpp -- Open MP directive lowering --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenMP.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/parse-tree.h"

#define TODO() llvm_unreachable("not yet implemented")

void Fortran::lower::genOpenMPConstruct(
    Fortran::lower::AbstractConverter &, Fortran::lower::pft::Evaluation &,
    const Fortran::parser::OpenMPConstruct &) {
  TODO();
}

void Fortran::lower::genOpenMPEndLoop(
    Fortran::lower::AbstractConverter &, Fortran::lower::pft::Evaluation &,
    const Fortran::parser::OmpEndLoopDirective &) {
  TODO();
}
