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

// template <class T> class polymorphic_allocator

// template <class P1, class P2, class U1, class U2>
// void polymorphic_allocator<T>::construct(pair<P1, P2>*, pair<U1, U2> &&)

#include <memory_resource>
#include <type_traits>
#include <utility>
#include <tuple>
#include <cassert>
#include <cstdlib>

#include "test_macros.h"
#include "test_std_memory_resource.h"
#include "uses_alloc_types.h"
#include "controlled_allocators.h"
#include "test_allocator.h"

template <class UA1, class UA2, class TT, class UU>
bool doTest(UsesAllocatorType TExpect, UsesAllocatorType UExpect, std::pair<TT, UU>&& p) {
  using P = std::pair<UA1, UA2>;
  TestResource R;
  std::pmr::memory_resource* M = &R;
  std::pmr::polymorphic_allocator<P> A(M);
  P* ptr  = A.allocate(2);
  P* ptr2 = ptr + 1;

  // UNDER TEST //
  A.construct(ptr, std::move(p));

  A.construct(ptr2,
              std::piecewise_construct,
              std::forward_as_tuple(std::forward<TT>(p.first)),
              std::forward_as_tuple(std::forward<UU>(p.second)));
  // ------- //

  bool tres = checkConstruct<TT&&>(ptr->first, TExpect, M) && checkConstructionEquiv(ptr->first, ptr2->first);

  bool ures = checkConstruct<UU&&>(ptr->second, UExpect, M) && checkConstructionEquiv(ptr->second, ptr2->second);

  A.destroy(ptr);
  A.destroy(ptr2);
  A.deallocate(ptr, 2);
  return tres && ures;
}

template <class Alloc, class TT, class UU>
void test_pmr_uses_allocator(std::pair<TT, UU>&& p) {
  {
    using T = NotUsesAllocator<Alloc, 1>;
    using U = NotUsesAllocator<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, std::move(p))));
  }
  {
    using T = UsesAllocatorV1<Alloc, 1>;
    using U = UsesAllocatorV2<Alloc, 1>;
    assert((doTest<T, U>(UA_AllocArg, UA_AllocLast, std::move(p))));
  }
  {
    using T = UsesAllocatorV2<Alloc, 1>;
    using U = UsesAllocatorV3<Alloc, 1>;
    assert((doTest<T, U>(UA_AllocLast, UA_AllocArg, std::move(p))));
  }
  {
    using T = UsesAllocatorV3<Alloc, 1>;
    using U = NotUsesAllocator<Alloc, 1>;
    assert((doTest<T, U>(UA_AllocArg, UA_None, std::move(p))));
  }
}

template <class Alloc, class TT, class UU>
void test_pmr_not_uses_allocator(std::pair<TT, UU>&& p) {
  {
    using T = NotUsesAllocator<Alloc, 1>;
    using U = NotUsesAllocator<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, std::move(p))));
  }
  {
    using T = UsesAllocatorV1<Alloc, 1>;
    using U = UsesAllocatorV2<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, std::move(p))));
  }
  {
    using T = UsesAllocatorV2<Alloc, 1>;
    using U = UsesAllocatorV3<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, std::move(p))));
  }
  {
    using T = UsesAllocatorV3<Alloc, 1>;
    using U = NotUsesAllocator<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, std::move(p))));
  }
}

int main(int, char**) {
  using PMR = std::pmr::memory_resource*;
  using PMA = std::pmr::polymorphic_allocator<char>;
  {
    int x = 42;
    int y = 42;
    std::pair<int&, int&&> p(x, std::move(y));
    test_pmr_not_uses_allocator<PMR>(std::move(p));
    test_pmr_uses_allocator<PMA>(std::move(p));
  }
  {
    int x = 42;
    int y = 42;
    std::pair<int&&, int&> p(std::move(x), y);
    test_pmr_not_uses_allocator<PMR>(std::move(p));
    test_pmr_uses_allocator<PMA>(std::move(p));
  }

  return 0;
}
