//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: stdlib=apple-libc++

// Test that using -pedantic-errors doesn't turn off availability annotations.
// This used to be the case because we used __has_extension(...) to enable the
// availability annotations, and -pedantic-errors changes the behavior of
// __has_extension(...) in an incompatible way.

// ADDITIONAL_COMPILE_FLAGS: -pedantic-errors

#include <__config>

#if !_LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS
#  error Availability annotations should be enabled on Apple platforms in the system configuration!
#endif
