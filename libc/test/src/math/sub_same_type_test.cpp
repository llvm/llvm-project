//===-- Unittests for fputil::generic::sub --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubTest.h"

#include "src/__support/FPUtil/generic/add_sub.h"
#include "src/__support/macros/properties/types.h"

#define SUB_FUNC(T) (LIBC_NAMESPACE::fputil::generic::sub<T, T>)

LIST_SUB_SAME_TYPE_TESTS(Double, double, double, SUB_FUNC(double))
LIST_SUB_SAME_TYPE_TESTS(Float, float, float, SUB_FUNC(float))
LIST_SUB_SAME_TYPE_TESTS(LongDouble, long double, long double,
                         SUB_FUNC(long double))
#ifdef LIBC_TYPES_HAS_FLOAT16
LIST_SUB_SAME_TYPE_TESTS(Float16, float16, float16, SUB_FUNC(float16))
#endif
#ifdef LIBC_TYPES_HAS_FLOAT128
LIST_SUB_SAME_TYPE_TESTS(Float128, float128, float128, SUB_FUNC(float128))
#endif
