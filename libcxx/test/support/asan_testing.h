//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_TESTING_H
#define ASAN_TESTING_H

#include "test_macros.h"
#include <vector>

#if TEST_HAS_FEATURE(address_sanitizer)
extern "C" int __sanitizer_verify_contiguous_container
     ( const void *beg, const void *mid, const void *end );

template <typename T, typename Alloc>
TEST_CONSTEXPR bool is_contiguous_container_asan_correct ( const std::vector<T, Alloc> &c )
{
    if (std::__libcpp_is_constant_evaluated())
        return true;
    if (std::is_same<Alloc, std::allocator<T> >::value && c.data() != NULL)
        return __sanitizer_verify_contiguous_container(
            c.data(), c.data() + c.size(), c.data() + c.capacity()) != 0;
    return true;
}
#else
template <typename T, typename Alloc>
TEST_CONSTEXPR bool is_contiguous_container_asan_correct ( const std::vector<T, Alloc> &)
{
    return true;
}
#endif // TEST_HAS_FEATURE(address_sanitizer)

#if TEST_HAS_FEATURE(address_sanitizer) && _LIBCPP_CLANG_VER >= 1600
extern "C" int __sanitizer_verify_double_ended_contiguous_container(
    const void* beg, const void* con_beg, const void* con_end, const void* end);
extern "C" bool __sanitizer_is_annotable(const void* address, const unsigned long size);
#include <deque>

template <class T, class Alloc>
TEST_CONSTEXPR bool is_double_ended_contiguous_container_asan_correct(const std::deque<T, Alloc>& c) {
  if (TEST_IS_CONSTANT_EVALUATED)
    return true;
  if (std::is_same<Alloc, std::allocator<T> >::value)
    return c.__verify_asan_annotations();
  return true;
}
#else
#  include <deque>
template <class T, class Alloc>
TEST_CONSTEXPR bool is_double_ended_contiguous_container_asan_correct(const std::deque<T, Alloc>&) {
  return true;
}
#endif

#endif // ASAN_TESTING_H
