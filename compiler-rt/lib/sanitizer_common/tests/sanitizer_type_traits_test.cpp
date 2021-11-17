//===-- sanitizer_type_traits_test.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_type_traits.h"
#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

using namespace __sanitizer;

TEST(SanitizerCommon, IsSame) {
  ASSERT_TRUE((is_same<unsigned, unsigned>::value));
  ASSERT_TRUE((is_same<uptr, uptr>::value));
  ASSERT_TRUE((is_same<sptr, sptr>::value));
  ASSERT_TRUE((is_same<const uptr, const uptr>::value));

  ASSERT_FALSE((is_same<unsigned, signed>::value));
  ASSERT_FALSE((is_same<uptr, sptr>::value));
  ASSERT_FALSE((is_same<uptr, const uptr>::value));
}

TEST(SanitizerCommon, Conditional) {
  ASSERT_TRUE((is_same<int, conditional<true, int, double>::type>::value));
  ASSERT_TRUE((is_same<double, conditional<false, int, double>::type>::value));
}
