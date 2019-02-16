//===-- sanitizer_type_traits_test.cc -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
