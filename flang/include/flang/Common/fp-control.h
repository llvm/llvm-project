//===-- include/flang/Common/fp-control.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FLANG_FENV_ACCESS_ON enables floating-point environment access in the
// enclosing scope, so that the compiler does not reorder or elide calls to
// fenv.h primitives (feraiseexcept, fesetround, fetestexcept, ...).  It also
// silences clang's -Wfenv-access diagnostic on those calls.
//
// Use as a statement at the top of a function body:
//
//   void f() {
//     FLANG_FENV_ACCESS_ON
//     feraiseexcept(FE_INVALID);
//   }

#ifndef FORTRAN_COMMON_FP_CONTROL_H_
#define FORTRAN_COMMON_FP_CONTROL_H_

#ifdef HAVE_STDC_FENV_ACCESS 
#define FLANG_FENV_ACCESS_ON _Pragma("STDC FENV_ACCESS ON")
#else
#define FLANG_FENV_ACCESS_ON
#endif

#endif // FORTRAN_COMMON_FP_CONTROL_H_
