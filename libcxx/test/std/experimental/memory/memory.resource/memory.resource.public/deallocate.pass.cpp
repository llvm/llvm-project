//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <experimental/memory_resource>

// UNSUPPORTED: c++03

//------------------------------------------------------------------------------
// TESTING void * memory_resource::deallocate(void *, size_t, size_t = max_align)
//
// Concerns:
//  A) 'memory_resource' contains a member 'deallocate' with the required
//     signature, including the default alignment parameter.
//  B) The return type of 'deallocate' is 'void'.
//  C) 'deallocate' is not marked as 'noexcept'.
//  D) Invoking 'deallocate' invokes 'do_deallocate' with the same arguments.


#include <experimental/memory_resource>
#include <type_traits>
#include <cstddef>
#include <cassert>

#include "test_memory_resource.h"

#include "test_macros.h"

using std::experimental::pmr::memory_resource;

int main(int, char**)
{
    NullResource R(42);
    auto& P = R.getController();
    memory_resource& M = R;
    {
        static_assert(
            std::is_same<decltype(M.deallocate(nullptr, 0, 0)), void>::value
          , "Must be void"
          );
        static_assert(
            std::is_same<decltype(M.deallocate(nullptr, 0)), void>::value
          , "Must be void"
          );
    }
    {
        static_assert(
            ! noexcept(M.deallocate(nullptr, 0, 0))
          , "Must not be noexcept."
          );
        static_assert(
            ! noexcept(M.deallocate(nullptr, 0))
          , "Must not be noexcept."
          );
    }
    {
        int s = 100;
        int a = 64;
        void* p = reinterpret_cast<void*>(640);
        M.deallocate(p, s, a);
        assert(P.dealloc_count == 1);
        assert(P.checkDealloc(p, s, a));

        s = 128;
        a = alignof(std::max_align_t);
        p = reinterpret_cast<void*>(12800);
        M.deallocate(p, s);
        assert(P.dealloc_count == 2);
        assert(P.checkDealloc(p, s, a));
    }

  return 0;
}
