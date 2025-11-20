//===-- Unittests for fputil::generic::add --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AddTest.h"

#include "src/__support/FPUtil/generic/add_sub.h"
#include "src/__support/macros/properties/types.h"

#define ADD_FUNC(T) (LIBC_NAMESPACE::fputil::generic::add<T, T>)

LIST_ADD_SAME_TYPE_TESTS(Double, double, double, ADD_FUNC(double))
LIST_ADD_SAME_TYPE_TESTS(Float, float, float, ADD_FUNC(float))
LIST_ADD_SAME_TYPE_TESTS(LongDouble, long double, long double,
                         ADD_FUNC(long double))
#ifdef LIBC_TYPES_HAS_FLOAT16
LIST_ADD_SAME_TYPE_TESTS(Float16, float16, float16, ADD_FUNC(float16))
#endif
#ifdef LIBC_TYPES_HAS_FLOAT128
LIST_ADD_SAME_TYPE_TESTS(Float128, float128, float128, ADD_FUNC(float128))
#endif
