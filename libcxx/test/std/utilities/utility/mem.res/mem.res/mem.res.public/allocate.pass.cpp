//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://github.com/llvm/llvm-project/issues/40340 is fixed
// UNSUPPORTED: availability-pmr-missing

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <memory_resource>

//------------------------------------------------------------------------------
// TESTING void *memory_resource::allocate(size_t, size_t = max_align)
//
// Concerns:
//  A) 'memory_resource' contains a member 'allocate' with the required
//     signature, including the default alignment parameter.
//  B) The return type of 'allocate' is 'void*'.
//  C) 'allocate' is not marked as 'noexcept'.
//  D) Invoking 'allocate' invokes 'do_allocate' with the same arguments.
//  E) If 'do_allocate' throws then 'allocate' propagates that exception.

#include <memory_resource>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"
#include "test_std_memory_resource.h"

int main(int, char**) {
  TestResource R(42);
  auto& P                      = R.getController();
  std::pmr::memory_resource& M = R;
  {
    static_assert(std::is_same<decltype(M.allocate(0, 1)), void*>::value, "Must be void*");
    static_assert(std::is_same<decltype(M.allocate(0)), void*>::value, "Must be void*");
  }
  {
    static_assert(!noexcept(M.allocate(0, 1)), "Must not be noexcept.");
    static_assert(!noexcept(M.allocate(0)), "Must not be noexcept.");
  }
  {
    int s   = 42;
    int a   = 64;
    void* p = M.allocate(s, a);
    assert(P.alloc_count == 1);
    assert(P.checkAlloc(p, s, a));

    s = 128;
    a = MaxAlignV;
    p = M.allocate(s);
    assert(P.alloc_count == 2);
    assert(P.checkAlloc(p, s, a));
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    TestResource R2;
    auto& P2                      = R2.getController();
    P2.throw_on_alloc             = true;
    std::pmr::memory_resource& M2 = R2;
    try {
      (void)M2.allocate(42);
      assert(false);
    } catch (TestException const&) {
      // do nothing.
    } catch (...) {
      assert(false);
    }
  }
#endif

  return 0;
}
