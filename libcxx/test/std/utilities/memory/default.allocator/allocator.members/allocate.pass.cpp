//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: clang-8
// UNSUPPORTED: gcc-8, gcc-9

// <memory>

// allocator:
// constexpr T* allocate(size_t n);

#include <memory>
#include <cassert>
#include <cstddef>       // for std::max_align_t

#include "test_macros.h"
#include "count_new.h"


#ifdef TEST_HAS_NO_ALIGNED_ALLOCATION
static const bool UsingAlignedNew = false;
#else
static const bool UsingAlignedNew = true;
#endif

#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
static const size_t MaxAligned = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#else
static const size_t MaxAligned = std::alignment_of<std::max_align_t>::value;
#endif

static const size_t OverAligned = MaxAligned * 2;


template <size_t Align>
struct TEST_ALIGNAS(Align) AlignedType {
  char data;
  static int constructed;
  AlignedType() { ++constructed; }
  AlignedType(AlignedType const&) { ++constructed; }
  ~AlignedType() { --constructed; }
};
template <size_t Align>
int AlignedType<Align>::constructed = 0;


template <size_t Align>
void test_aligned() {
  typedef AlignedType<Align> T;
  T::constructed = 0;
  globalMemCounter.reset();
  std::allocator<T> a;
  const bool IsOverAlignedType = Align > MaxAligned;
  const bool ExpectAligned = IsOverAlignedType && UsingAlignedNew;
  {
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(T::constructed == 0);
    globalMemCounter.last_new_size = 0;
    globalMemCounter.last_new_align = 0;
    T* ap = a.allocate(3);
    DoNotOptimize(ap);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(globalMemCounter.checkNewCalledEq(1));
    assert(globalMemCounter.checkAlignedNewCalledEq(ExpectAligned));
    assert(globalMemCounter.checkLastNewSizeEq(3 * sizeof(T)));
    assert(globalMemCounter.checkLastNewAlignEq(ExpectAligned ? Align : 0));
    assert(T::constructed == 0);
    globalMemCounter.last_delete_align = 0;
    a.deallocate(ap, 3);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(globalMemCounter.checkDeleteCalledEq(1));
    assert(globalMemCounter.checkAlignedDeleteCalledEq(ExpectAligned));
    assert(globalMemCounter.checkLastDeleteAlignEq(ExpectAligned ? Align : 0));
    assert(T::constructed == 0);
  }
}

#if TEST_STD_VER > 17
template <size_t Align>
constexpr bool test_aligned_constexpr() {
    typedef AlignedType<Align> T;
    std::allocator<T> a;
    T* ap = a.allocate(3);
    a.deallocate(ap, 3);

    return true;
}
#endif

int main(int, char**) {
    test_aligned<1>();
    test_aligned<2>();
    test_aligned<4>();
    test_aligned<8>();
    test_aligned<16>();
    test_aligned<MaxAligned>();
    test_aligned<OverAligned>();
    test_aligned<OverAligned * 2>();

#if TEST_STD_VER > 17
    static_assert(test_aligned_constexpr<1>());
    static_assert(test_aligned_constexpr<2>());
    static_assert(test_aligned_constexpr<4>());
    static_assert(test_aligned_constexpr<8>());
    static_assert(test_aligned_constexpr<16>());
    static_assert(test_aligned_constexpr<MaxAligned>());
    static_assert(test_aligned_constexpr<OverAligned>());
    static_assert(test_aligned_constexpr<OverAligned * 2>());
#endif

  return 0;
}
