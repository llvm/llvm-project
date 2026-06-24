//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>
//
// constexpr reference operator[](size_type idx) const;

// Make sure that accessing a span out-of-bounds triggers an assertion in an
// Objective-C++ translation unit.

// REQUIRES: objective-c++
// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -lobjc

#include <span>

#include "check_assertion.h"

#import <objc/NSObject.h>

@interface Wrapper : NSObject {
  int _array[3];
}
- (std::span<int>)dynamicSpan;
- (std::span<int, 3>)staticSpan;
@end

@implementation Wrapper
- (std::span<int>)dynamicSpan { return std::span<int>(_array, 3); }
- (std::span<int, 3>)staticSpan { return std::span<int, 3>(_array, 3); }
@end

int main(int, char**) {
  Wrapper* wrapper = [[Wrapper alloc] init];

  TEST_LIBCPP_ASSERT_FAILURE([wrapper dynamicSpan][3], "span<T>::operator[](index): index out of range");
  TEST_LIBCPP_ASSERT_FAILURE([wrapper staticSpan][3], "span<T, N>::operator[](index): index out of range");

  return 0;
}
