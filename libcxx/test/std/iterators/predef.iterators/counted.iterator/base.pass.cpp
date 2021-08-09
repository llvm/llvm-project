//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// constexpr const I& base() const &;
// constexpr I base() &&;

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

struct InputOrOutputArchetype {
  using difference_type = int;

  int *ptr;

  int operator*() { return *ptr; }
  void operator++(int) { ++ptr; }
  InputOrOutputArchetype& operator++() { ++ptr; return *this; }
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(iter.base().base() == buffer);
    assert(std::move(iter).base().base() == buffer);

    ASSERT_SAME_TYPE(decltype(iter.base()), const cpp20_input_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(std::move(iter).base()), cpp20_input_iterator<int*>);
  }

  {
    std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    assert(iter.base() == forward_iterator<int*>{buffer});
    assert(std::move(iter).base() == forward_iterator<int*>{buffer});

    ASSERT_SAME_TYPE(decltype(iter.base()), const forward_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(std::move(iter).base()), forward_iterator<int*>);
  }

  {
    std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    assert(iter.base() == contiguous_iterator<int*>{buffer});
    assert(std::move(iter).base() == contiguous_iterator<int*>{buffer});

    ASSERT_SAME_TYPE(decltype(iter.base()), const contiguous_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(std::move(iter).base()), contiguous_iterator<int*>);
  }

  {
    std::counted_iterator iter(InputOrOutputArchetype{buffer}, 6);
    assert(iter.base().ptr == buffer);
    assert(std::move(iter).base().ptr == buffer);

    ASSERT_SAME_TYPE(decltype(iter.base()), const InputOrOutputArchetype&);
    ASSERT_SAME_TYPE(decltype(std::move(iter).base()), InputOrOutputArchetype);
  }

  {
    const std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(iter.base().base() == buffer);
    assert(std::move(iter).base().base() == buffer);

    ASSERT_SAME_TYPE(decltype(iter.base()), const cpp20_input_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(std::move(iter).base()), const cpp20_input_iterator<int*>&);
  }

  {
    const std::counted_iterator iter(forward_iterator<int*>{buffer}, 7);
    assert(iter.base() == forward_iterator<int*>{buffer});
    assert(std::move(iter).base() == forward_iterator<int*>{buffer});

    ASSERT_SAME_TYPE(decltype(iter.base()), const forward_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(std::move(iter).base()), const forward_iterator<int*>&);
  }

  {
    const std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 6);
    assert(iter.base() == contiguous_iterator<int*>{buffer});
    assert(std::move(iter).base() == contiguous_iterator<int*>{buffer});

    ASSERT_SAME_TYPE(decltype(iter.base()), const contiguous_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(std::move(iter).base()), const contiguous_iterator<int*>&);
  }

  {
    const std::counted_iterator iter(InputOrOutputArchetype{buffer}, 6);
    assert(iter.base().ptr == buffer);
    assert(std::move(iter).base().ptr == buffer);

    ASSERT_SAME_TYPE(decltype(iter.base()), const InputOrOutputArchetype&);
    ASSERT_SAME_TYPE(decltype(std::move(iter).base()), const InputOrOutputArchetype&);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
