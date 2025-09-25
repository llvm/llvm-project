//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <iterator>

// __bounded_iter<_Iter>

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// Verify that __bounded_iter does not accept non-contiguous iterators as determined by __libcpp_is_contiguous_iterator.
// static_assert should be used, see https://llvm.org/PR115002.
// __wrap_iter cannot be so handled because it may directly wrap user-defined fancy pointers in libc++'s vector.

#include <deque>
#include <vector>
#include <array>

// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}Only contiguous iterators can be adapted by __bounded_iter.}}
std::__bounded_iter<std::deque<int>::iterator> bounded_iter;
// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}Only contiguous iterators can be adapted by __static_bounded_iter.}}
std::__static_bounded_iter<std::deque<int>::iterator, 42> statically_bounded_iter;
