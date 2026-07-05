//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// This test ensures that std::tuple is lazy when it comes to checking whether
// the elements it is assigned from can be used to assign to the types in
// the tuple.

#include <tuple>
#include <array>

template <bool Enable, class ...Class>
constexpr typename std::enable_if<Enable, bool>::type BlowUp() {
  static_assert(Enable && sizeof...(Class) != sizeof...(Class), "");
  return true;
}

template<class T>
struct Fail {
  static_assert(sizeof(T) != sizeof(T), "");
  using type = void;
};

struct NoAssign {
  NoAssign() = default;
  NoAssign(NoAssign const&) = default;
  template <class T, class = typename std::enable_if<sizeof(T) != sizeof(T)>::type>
  NoAssign& operator=(T) { return *this; }
};

template <int>
struct DieOnAssign {
  DieOnAssign() = default;
  template <class T, class X = typename std::enable_if<!std::is_same<T, DieOnAssign>::value>::type,
                     class = typename Fail<X>::type>
  DieOnAssign& operator=(T) {
    return *this;
  }
};

void test_arity_checks() {
  {
    using T = std::tuple<int, DieOnAssign<0>, int>;
    using P = std::pair<int, int>;
    static_assert(!std::is_assignable<T&, P const&>::value, "");
  }
  {
    using T = std::tuple<int, int, DieOnAssign<1> >;
    using A = std::array<int, 1>;
    static_assert(!std::is_assignable<T&, A const&>::value, "");
  }
}

void test_assignability_checks() {
  {
    using T1 = std::tuple<int, NoAssign, DieOnAssign<2> >;
    using T2 = std::tuple<long, long, long>;
    static_assert(!std::is_assignable<T1&, T2 const&>::value, "");
  }
  {
    using T1 = std::tuple<NoAssign, DieOnAssign<3> >;
    using T2 = std::pair<long, double>;
    static_assert(!std::is_assignable<T1&, T2 const&>::value, "");
  }
}

int main(int, char**) {
  test_arity_checks();
  test_assignability_checks();
  return 0;
}
