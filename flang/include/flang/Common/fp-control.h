//===-- include/flang/Common/fp-control.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FLANG_FP_TRAP_ON enables floating-point exception access in the
// enclosing scope.  It silences clang's -Wfenv-access warning on calls to
// fenv.h primitives (feraiseexcept, fesetround, fetestexcept, ...).
//
// Use as a statement at the top of a function body:
//
//   void f() {
//     FLANG_FP_TRAP_ON
//     feraiseexcept(FE_INVALID);
//   }
//

#ifndef FORTRAN_COMMON_FP_CONTROL_H_
#define FORTRAN_COMMON_FP_CONTROL_H_

#if defined(__clang__) && (__clang_major__ >= 10)
// Clang >= 10 supports `#pragma clang fp exceptions(maytrap)`, which is the
// local-scope equivalent of `-ffp-exception-behavior=maytrap` and is what the
// -Wfenv-access diagnostic recommends.
#define FLANG_FP_TRAP_ON _Pragma("clang fp exceptions(maytrap)")
#else
// Portable fallback for GCC, MSVC, or older clang.
#define FLANG_FP_TRAP_ON _Pragma("STDC FENV_ACCESS ON")
#endif

#endif // FORTRAN_COMMON_FP_CONTROL_H_
