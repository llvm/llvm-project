//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <memory_resource>

// template <class T> class polymorphic_allocator

// template <class P1, class P2, class U1, class U2>
// void polymorphic_allocator<T>::construct(pair<P1, P2>*, pair<U1, U2> const&)

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
bool doTest(UsesAllocatorType TExpect, UsesAllocatorType UExpect, std::pair<TT, UU> const& p) {
  using P = std::pair<UA1, UA2>;
  TestResource R;
  std::pmr::memory_resource* M = &R;
  std::pmr::polymorphic_allocator<P> A(M);
  P* ptr  = (P*)std::malloc(sizeof(P));
  P* ptr2 = (P*)std::malloc(sizeof(P));

  // UNDER TEST //
  A.construct(ptr, p);

  A.construct(ptr2, std::piecewise_construct, std::forward_as_tuple(p.first), std::forward_as_tuple(p.second));
  // ------- //

  bool tres =
      checkConstruct<decltype((p.first))>(ptr->first, TExpect, M) && checkConstructionEquiv(ptr->first, ptr2->first);

  bool ures = checkConstruct<decltype((p.second))>(ptr->second, UExpect, M) &&
              checkConstructionEquiv(ptr->second, ptr2->second);

  A.destroy(ptr);
  std::free(ptr);
  A.destroy(ptr2);
  std::free(ptr2);
  return tres && ures;
}

template <class Alloc, class TT, class UU>
void test_pmr_uses_allocator(std::pair<TT, UU> const& p) {
  {
    using T = NotUsesAllocator<Alloc, 1>;
    using U = NotUsesAllocator<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, p)));
  }
  {
    using T = UsesAllocatorV1<Alloc, 1>;
    using U = UsesAllocatorV2<Alloc, 1>;
    assert((doTest<T, U>(UA_AllocArg, UA_AllocLast, p)));
  }
  {
    using T = UsesAllocatorV2<Alloc, 1>;
    using U = UsesAllocatorV3<Alloc, 1>;
    assert((doTest<T, U>(UA_AllocLast, UA_AllocArg, p)));
  }
  {
    using T = UsesAllocatorV3<Alloc, 1>;
    using U = NotUsesAllocator<Alloc, 1>;
    assert((doTest<T, U>(UA_AllocArg, UA_None, p)));
  }
}

template <class Alloc, class TT, class UU>
void test_pmr_not_uses_allocator(std::pair<TT, UU> const& p) {
  {
    using T = NotUsesAllocator<Alloc, 1>;
    using U = NotUsesAllocator<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, p)));
  }
  {
    using T = UsesAllocatorV1<Alloc, 1>;
    using U = UsesAllocatorV2<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, p)));
  }
  {
    using T = UsesAllocatorV2<Alloc, 1>;
    using U = UsesAllocatorV3<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, p)));
  }
  {
    using T = UsesAllocatorV3<Alloc, 1>;
    using U = NotUsesAllocator<Alloc, 1>;
    assert((doTest<T, U>(UA_None, UA_None, p)));
  }
}

template <class Tp>
struct Print;

int main(int, char**) {
  using PMR = std::pmr::memory_resource*;
  using PMA = std::pmr::polymorphic_allocator<char>;
  {
    int x = 42;
    int y = 42;
    const std::pair<int, int&> p(x, y);
    test_pmr_not_uses_allocator<PMR>(p);
    test_pmr_uses_allocator<PMA>(p);
  }
  {
    int x = 42;
    int y = 42;
    const std::pair<int&, int&&> p(x, std::move(y));
    test_pmr_not_uses_allocator<PMR>(p);
    test_pmr_uses_allocator<PMA>(p);
  }

  return 0;
}
