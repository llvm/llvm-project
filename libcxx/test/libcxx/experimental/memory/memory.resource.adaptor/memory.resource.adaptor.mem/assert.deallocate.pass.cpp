//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <experimental/memory_resource>

// template <class T> class polymorphic_allocator

// T* polymorphic_allocator<T>::deallocate(T*, size_t size)

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "check_assertion.h"
#include "test_memory_resource.h"

namespace ex = std::experimental::pmr;

int main(int, char**) {
    using Alloc = NullAllocator<char>;

    AllocController P;
    ex::resource_adaptor<Alloc> r(Alloc{P});
    ex::memory_resource & m1 = r;

    std::size_t maxSize = std::numeric_limits<std::size_t>::max()
                            - alignof(std::max_align_t);

    m1.deallocate(nullptr, maxSize); // no assertion
    TEST_LIBCPP_ASSERT_FAILURE(m1.deallocate(nullptr, maxSize + 1), "do_deallocate called for size which exceeds the maximum allocation size");

    return 0;
}
