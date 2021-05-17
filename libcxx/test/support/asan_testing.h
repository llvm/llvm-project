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

#if TEST_HAS_FEATURE(address_sanitizer)
extern "C" int __sanitizer_verify_contiguous_container
     ( const void *beg, const void *mid, const void *end );

template <typename T, typename Alloc>
bool is_contiguous_container_asan_correct ( const std::vector<T, Alloc> &c )
{
    if ( std::is_same<Alloc, std::allocator<T> >::value && c.data() != NULL)
        return __sanitizer_verify_contiguous_container (
            c.data(), c.data() + c.size(), c.data() + c.capacity()) != 0;
    return true;
}

#else
template <typename T, typename Alloc>
bool is_contiguous_container_asan_correct ( const std::vector<T, Alloc> &)
{
    return true;
}
#endif


#endif // ASAN_TESTING_H
