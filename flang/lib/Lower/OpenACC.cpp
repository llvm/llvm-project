//===-- OpenMP.cpp -- OpenACC directive lowering --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenACC.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

#define TODO() llvm_unreachable("not yet implemented")

void Fortran::lower::genOpenACCConstruct(
    Fortran::lower::AbstractConverter &absConv,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenACCConstruct &accConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenACCBlockConstruct &blockConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCCombinedConstruct
                  &combinedConstruct) { TODO(); },
          [&](const Fortran::parser::OpenACCLoopConstruct &loopConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCStandaloneConstruct
                  &standaloneConstruct) { TODO(); },
          [&](const Fortran::parser::OpenACCRoutineConstruct
                  &routineConstruct) { TODO(); },
          [&](const Fortran::parser::OpenACCCacheConstruct &cacheConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCWaitConstruct &waitConstruct) {
            TODO();
          },
          [&](const Fortran::parser::OpenACCAtomicConstruct &atomicConstruct) {
            TODO();
          },
      },
      accConstruct.u);
}
