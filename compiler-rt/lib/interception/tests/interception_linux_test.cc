//===-- interception_linux_test.cc ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
// Tests for interception_linux.h.
//
//===----------------------------------------------------------------------===//
#include "interception/interception.h"

#include "gtest/gtest.h"

// Too slow for debug build
#if !SANITIZER_DEBUG
#if SANITIZER_LINUX

static int InterceptorFunctionCalled;

DECLARE_REAL(int, isdigit, int);

INTERCEPTOR(int, isdigit, int d) {
  ++InterceptorFunctionCalled;
  return d >= '0' && d <= '9';
}

namespace __interception {

TEST(Interception, GetRealFunctionAddress) {
  uptr expected_malloc_address = (uptr)(void*)&malloc;
  uptr malloc_address = 0;
  EXPECT_TRUE(GetRealFunctionAddress("malloc", &malloc_address, 0, 0));
  EXPECT_EQ(expected_malloc_address, malloc_address);

  uptr dummy_address = 0;
  EXPECT_TRUE(
      GetRealFunctionAddress("dummy_doesnt_exist__", &dummy_address, 0, 0));
  EXPECT_EQ(0U, dummy_address);
}

TEST(Interception, Basic) {
  ASSERT_TRUE(INTERCEPT_FUNCTION(isdigit));

  // After interception, the counter should be incremented.
  InterceptorFunctionCalled = 0;
  EXPECT_NE(0, isdigit('1'));
  EXPECT_EQ(1, InterceptorFunctionCalled);
  EXPECT_EQ(0, isdigit('a'));
  EXPECT_EQ(2, InterceptorFunctionCalled);

  // Calling the REAL function should not affect the counter.
  InterceptorFunctionCalled = 0;
  EXPECT_NE(0, REAL(isdigit)('1'));
  EXPECT_EQ(0, REAL(isdigit)('a'));
  EXPECT_EQ(0, InterceptorFunctionCalled);
}

}  // namespace __interception

#endif  // SANITIZER_LINUX
#endif  // #if !SANITIZER_DEBUG
