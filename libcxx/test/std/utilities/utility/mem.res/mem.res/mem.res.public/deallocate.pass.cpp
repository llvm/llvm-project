//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <memory_resource>

//------------------------------------------------------------------------------
// TESTING void * memory_resource::deallocate(void *, size_t, size_t = max_align)
//
// Concerns:
//  A) 'memory_resource' contains a member 'deallocate' with the required
//     signature, including the default alignment parameter.
//  B) The return type of 'deallocate' is 'void'.
//  C) 'deallocate' is not marked as 'noexcept'.
//  D) Invoking 'deallocate' invokes 'do_deallocate' with the same arguments.

#include <memory_resource>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_std_memory_resource.h"

int main(int, char**) {
  NullResource R(42);
  auto& P                      = R.getController();
  std::pmr::memory_resource& M = R;
  {
    ASSERT_SAME_TYPE(decltype(M.deallocate(std::declval<int*>(), 0, 0)), void);
    ASSERT_SAME_TYPE(decltype(M.deallocate(std::declval<int*>(), 0)), void);
  }
  {
    static_assert(!noexcept(M.deallocate(std::declval<int*>(), 0, 0)));
    static_assert(!noexcept(M.deallocate(std::declval<int*>(), 0)));
  }
  {
    int s   = 100;
    int a   = 64;
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
