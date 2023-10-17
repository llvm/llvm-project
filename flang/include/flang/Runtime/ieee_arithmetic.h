#if 0 /*===-- include/flang/Runtime/ieee_arithmetic.h -------------------===*/
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*/
#endif
#if 0
This header can be included into both Fortran and C/C++.

Fortran 2018 Clause 17.2 Fortran intrinsic module ieee_exceptions values.
#endif

#ifndef FORTRAN_RUNTIME_IEEE_ARITHMETIC_H_
#define FORTRAN_RUNTIME_IEEE_ARITHMETIC_H_

#if 0
ieee_class_type values
The sequence is that of f18 clause 17.2p3, but nothing depends on that.
#endif
#define _FORTRAN_RUNTIME_IEEE_SIGNALING_NAN 1
#define _FORTRAN_RUNTIME_IEEE_QUIET_NAN 2
#define _FORTRAN_RUNTIME_IEEE_NEGATIVE_INF 3
#define _FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL 4
#define _FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL 5
#define _FORTRAN_RUNTIME_IEEE_NEGATIVE_ZERO 6
#define _FORTRAN_RUNTIME_IEEE_POSITIVE_ZERO 7
#define _FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL 8
#define _FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL 9
#define _FORTRAN_RUNTIME_IEEE_POSITIVE_INF 10
#define _FORTRAN_RUNTIME_IEEE_OTHER_VALUE 11

#if 0
ieee_round_type values
The values are those of the llvm.get.rounding instrinsic, which is assumed by
intrinsic module procedures ieee_get_rounding_mode, ieee_set_rounding_mode,
and ieee_support_rounding.
#endif
#define _FORTRAN_RUNTIME_IEEE_TO_ZERO 0
#define _FORTRAN_RUNTIME_IEEE_NEAREST 1
#define _FORTRAN_RUNTIME_IEEE_UP 2
#define _FORTRAN_RUNTIME_IEEE_DOWN 3
#define _FORTRAN_RUNTIME_IEEE_AWAY 4
#define _FORTRAN_RUNTIME_IEEE_OTHER 5

#endif
