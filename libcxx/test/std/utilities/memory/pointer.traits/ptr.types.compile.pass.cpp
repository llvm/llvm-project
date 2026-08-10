//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T>
// struct pointer_traits<T*>
// {
//     using pointer = T*;
//     using element_type = T;
//     using difference_type = ptrdiff_t;
//     template <class U> using rebind = U*;
// };

#include <memory>
#include <cstddef>

#include "test_macros.h"

void f() {
  {
    using Ptr = int*;

    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::element_type, int);
    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::pointer, Ptr);
    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::difference_type, std::ptrdiff_t);
#if TEST_STD_VER >= 11
    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::rebind<long>, long*);
#else
    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::rebind<long>::other, long*);
#endif
  }

  {
    using Ptr = const int*;

    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::element_type, const int);
    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::pointer, Ptr);
    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::difference_type, std::ptrdiff_t);
#if TEST_STD_VER >= 11
    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::rebind<long>, long*);
#else
    ASSERT_SAME_TYPE(std::pointer_traits<Ptr>::rebind<long>::other, long*);
#endif
  }
}
