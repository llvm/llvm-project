//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// void swap(vector& x);

// class reference;
// friend void swap(reference x, reference y);
// friend void swap(reference x, bool&);
// friend void swap(bool& x, reference y);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <vector>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class T, class U, class = void>
struct can_qualified_std_swap_with : std::false_type {};
template <class T, class U>
struct can_qualified_std_swap_with<T, U, decltype((void)std::swap(std::declval<T>(), std::declval<U>()))>
    : std::true_type {};

template <class T>
struct can_qualified_std_swap : can_qualified_std_swap_with<T&, T&>::type {};

namespace adl_only {
void swap();

template <class T, class U, class = void>
struct can_swap_with : std::false_type {};
template <class T, class U>
struct can_swap_with<T, U, decltype((void)swap(std::declval<T>(), std::declval<U>()))> : std::true_type {};

template <class T>
struct can_swap : can_swap_with<T&, T&>::type {};
} // namespace adl_only

TEST_CONSTEXPR_CXX20 void test_vector_bool_swap() {
  {
    typedef std::vector<bool> VB;
    static_assert(can_qualified_std_swap<VB>::value, "");
    static_assert(adl_only::can_swap<VB>::value, "");

    std::vector<bool> v1(100);
    std::vector<bool> v2(200);
    v1.swap(v2);
    assert(v1.size() == 200);
    assert(v1.capacity() >= 200);
    assert(v2.size() == 100);
    assert(v2.capacity() >= 100);
  }
  {
    typedef test_allocator<bool> A;
    typedef std::vector<bool, A> VB;
    static_assert(can_qualified_std_swap<VB>::value, "");
    static_assert(adl_only::can_swap<VB>::value, "");

    std::vector<bool, A> v1(100, true, A(1, 1));
    std::vector<bool, A> v2(200, false, A(1, 2));
    swap(v1, v2);
    assert(v1.size() == 200);
    assert(v1.capacity() >= 200);
    assert(v2.size() == 100);
    assert(v2.capacity() >= 100);
    assert(v1.get_allocator().get_id() == 1);
    assert(v2.get_allocator().get_id() == 2);
  }
  {
    typedef other_allocator<bool> A;
    typedef std::vector<bool, A> VB;
    static_assert(can_qualified_std_swap<VB>::value, "");
    static_assert(adl_only::can_swap<VB>::value, "");

    std::vector<bool, A> v1(100, true, A(1));
    std::vector<bool, A> v2(200, false, A(2));
    swap(v1, v2);
    assert(v1.size() == 200);
    assert(v1.capacity() >= 200);
    assert(v2.size() == 100);
    assert(v2.capacity() >= 100);
    assert(v1.get_allocator() == A(2));
    assert(v2.get_allocator() == A(1));
  }
#if TEST_STD_VER >= 11
  {
    using A  = min_allocator<bool>;
    using VB = std::vector<bool, A>;
    static_assert(can_qualified_std_swap<VB>::value, "");
    static_assert(adl_only::can_swap<VB>::value, "");

    std::vector<bool, min_allocator<bool>> v1(100);
    std::vector<bool, min_allocator<bool>> v2(200);
    v1.swap(v2);
    assert(v1.size() == 200);
    assert(v1.capacity() >= 200);
    assert(v2.size() == 100);
    assert(v2.capacity() >= 100);
  }
  {
    using A = min_allocator<bool>;
    std::vector<bool, A> v1(100, true, A());
    std::vector<bool, A> v2(200, false, A());
    swap(v1, v2);
    assert(v1.size() == 200);
    assert(v1.capacity() >= 200);
    assert(v2.size() == 100);
    assert(v2.capacity() >= 100);
    assert(v1.get_allocator() == A());
    assert(v2.get_allocator() == A());
  }
#endif
}

TEST_CONSTEXPR_CXX20 void test_vector_bool_reference_swap() {
  { // Test that only homogeneous vector<bool, A>::reference swap is supported.
    typedef std::vector<bool>::reference VBRef1;
    typedef std::vector<bool, test_allocator<bool> >::reference VBRef2;
    static_assert(can_qualified_std_swap_with<VBRef1, VBRef2>::value == std::is_same<VBRef1, VBRef2>::value, "");
    static_assert(adl_only::can_swap_with<VBRef1, VBRef2>::value == std::is_same<VBRef1, VBRef2>::value, "");
  }
  {
    typedef std::vector<bool>::reference VBRef;
    static_assert(can_qualified_std_swap<VBRef>::value, "");
    static_assert(!can_qualified_std_swap_with<VBRef, VBRef>::value, "");
    static_assert(!can_qualified_std_swap_with<VBRef, bool&>::value, "");
    static_assert(!can_qualified_std_swap_with<bool&, VBRef>::value, "");
    static_assert(adl_only::can_swap<VBRef>::value, "");
    static_assert(adl_only::can_swap_with<VBRef, VBRef>::value, "");
    static_assert(adl_only::can_swap_with<VBRef, bool&>::value, "");
    static_assert(adl_only::can_swap_with<bool&, VBRef>::value, "");

    using std::swap;

    std::vector<bool> v(2);
    VBRef r1 = v[0];
    VBRef r2 = v[1];
    r1       = true;

    swap(r1, r2);
    assert(v[0] == false);
    assert(v[1] == true);

    bool b1 = true;
    swap(r1, b1);
    assert(v[0] == true);
    assert(b1 == false);

    swap(b1, r1);
    assert(v[0] == false);
    assert(b1 == true);
  }
#if TEST_STD_VER >= 11
  {
    using VBRef = std::vector<bool, min_allocator<bool>>::reference;
    static_assert(can_qualified_std_swap<VBRef>::value, "");
    static_assert(!can_qualified_std_swap_with<VBRef, VBRef>::value, "");
    static_assert(!can_qualified_std_swap_with<VBRef, bool&>::value, "");
    static_assert(!can_qualified_std_swap_with<bool&, VBRef>::value, "");
    static_assert(adl_only::can_swap<VBRef>::value, "");
    static_assert(adl_only::can_swap_with<VBRef, VBRef>::value, "");
    static_assert(adl_only::can_swap_with<VBRef, bool&>::value, "");
    static_assert(adl_only::can_swap_with<bool&, VBRef>::value, "");

    using std::swap;

    std::vector<bool, min_allocator<bool>> v(2);
    VBRef r1 = v[0];
    VBRef r2 = v[1];
    r1       = true;

    swap(r1, r2);
    assert(v[0] == false);
    assert(v[1] == true);

    bool b1 = true;
    swap(r1, b1);
    assert(v[0] == true);
    assert(b1 == false);

    swap(b1, r1);
    assert(v[0] == false);
    assert(b1 == true);
  }
#endif
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test_vector_bool_swap();
  test_vector_bool_reference_swap();
  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
