//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// bool operator==(array<T, N> const&, array<T, N> const&);
// bool operator!=(array<T, N> const&, array<T, N> const&);
// bool operator<(array<T, N> const&, array<T, N> const&);
// bool operator<=(array<T, N> const&, array<T, N> const&);
// bool operator>(array<T, N> const&, array<T, N> const&);
// bool operator>=(array<T, N> const&, array<T, N> const&);


#include <array>
#include <vector>
#include <cassert>

#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

template <class Array>
void test_compare(const Array& LHS, const Array& RHS) {
  typedef std::vector<typename Array::value_type> Vector;
  const Vector LHSV(LHS.begin(), LHS.end());
  const Vector RHSV(RHS.begin(), RHS.end());
  assert((LHS == RHS) == (LHSV == RHSV));
  assert((LHS != RHS) == (LHSV != RHSV));
  assert((LHS < RHS) == (LHSV < RHSV));
  assert((LHS <= RHS) == (LHSV <= RHSV));
  assert((LHS > RHS) == (LHSV > RHSV));
  assert((LHS >= RHS) == (LHSV >= RHSV));
}

int main()
{
  {
    typedef int T;
    typedef std::array<T, 3> C;
    C c1 = {1, 2, 3};
    C c2 = {1, 2, 3};
    C c3 = {3, 2, 1};
    C c4 = {1, 2, 1};
    test_compare(c1, c2);
    test_compare(c1, c3);
    test_compare(c1, c4);
  }
  {
    typedef int T;
    typedef std::array<T, 0> C;
    C c1 = {};
    C c2 = {};
    test_compare(c1, c2);
  }
}
