//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void assign(size_type n, const_reference v);

#include <vector>
#include <algorithm>
#include <cassert>
#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"
#include "test_iterators.h"
#if TEST_STD_VER >= 11
#include "emplace_constructible.h"
#include "container_test_types.h"
#endif


void test_emplaceable_concept() {
#if TEST_STD_VER >= 11
  int arr1[] = {42};
  int arr2[] = {1, 101, 42};
  {
    using T = EmplaceConstructibleMoveableAndAssignable<int>;
    using It = forward_iterator<int*>;
    {
      std::vector<T> v;
      v.assign(It(arr1), It(std::end(arr1)));
      assert(v[0].value == 42);
    }
    {
      std::vector<T> v;
      v.assign(It(arr2), It(std::end(arr2)));
      assert(v[0].value == 1);
      assert(v[1].value == 101);
      assert(v[2].value == 42);
    }
  }
  {
    using T = EmplaceConstructibleMoveableAndAssignable<int>;
    using It = cpp17_input_iterator<int*>;
    {
      std::vector<T> v;
      v.assign(It(arr1), It(std::end(arr1)));
      assert(v[0].copied == 0);
      assert(v[0].value == 42);
    }
    {
      std::vector<T> v;
      v.assign(It(arr2), It(std::end(arr2)));
      //assert(v[0].copied == 0);
      assert(v[0].value == 1);
      //assert(v[1].copied == 0);
      assert(v[1].value == 101);
      assert(v[2].copied == 0);
      assert(v[2].value == 42);
    }
  }
#endif
}



int main(int, char**)
{
    test_emplaceable_concept();

  return 0;
}
