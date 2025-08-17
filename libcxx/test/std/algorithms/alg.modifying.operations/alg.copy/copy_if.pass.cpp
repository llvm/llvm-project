//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter,
//          Predicate<auto, InIter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr OutIter   // constexpr after C++17
//   copy_if(InIter first, InIter last, OutIter result, Pred pred);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

struct Pred {
  TEST_CONSTEXPR_CXX14 bool operator()(int i) { return i % 3 == 0; }
};

template <class InIter>
struct TestOutIters {
  template <class OutIter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    const unsigned N = 1000;
    int ia[N]        = {};
    for (unsigned i = 0; i < N; ++i)
      ia[i] = i;
    int ib[N] = {0};

    OutIter r = std::copy_if(InIter(ia), InIter(ia + N), OutIter(ib), Pred());
    assert(base(r) == ib + N / 3 + 1);
    for (unsigned i = 0; i < N / 3 + 1; ++i)
      assert(ib[i] % 3 == 0);
  }
};

struct TestInIters {
  template <class InIter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(
        types::concatenate_t<types::cpp17_input_iterator_list<int*>, types::type_list<cpp17_output_iterator<int*> > >(),
        TestOutIters<InIter>());
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::cpp17_input_iterator_list<const int*>(), TestInIters());
  return true;
}

int main(int, char**) {
  test();

#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
