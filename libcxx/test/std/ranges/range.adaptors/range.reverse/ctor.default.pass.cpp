//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// reverse_view() requires default_­initializable<V> = default;

#include <ranges>
#include <cassert>

#include "types.h"

enum CtorKind { DefaultCtor, PtrCtor };
template<CtorKind CK>
struct BidirRangeWith : std::ranges::view_base {
  int *ptr_ = nullptr;

  constexpr BidirRangeWith() requires (CK == DefaultCtor) = default;
  constexpr BidirRangeWith(int *ptr);

  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{ptr_}; }
  constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{ptr_}; }
  constexpr bidirectional_iterator<int*> end() { return bidirectional_iterator<int*>{ptr_ + 8}; }
  constexpr bidirectional_iterator<const int*> end() const { return bidirectional_iterator<const int*>{ptr_ + 8}; }
};

constexpr bool test() {
  {
    static_assert( std::default_initializable<std::ranges::reverse_view<BidirRangeWith<DefaultCtor>>>);
    static_assert(!std::default_initializable<std::ranges::reverse_view<BidirRangeWith<PtrCtor>>>);
  }

  {
    std::ranges::reverse_view<BidirRangeWith<DefaultCtor>> rev;
    assert(rev.base().ptr_ == nullptr);
  }
  {
    const std::ranges::reverse_view<BidirRangeWith<DefaultCtor>> rev;
    assert(rev.base().ptr_ == nullptr);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

