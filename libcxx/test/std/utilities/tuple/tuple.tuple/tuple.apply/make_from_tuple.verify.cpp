//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <tuple>

// template <class T, class Tuple> constexpr T make_from_tuple(Tuple&&);

#include <tuple>

int main() {
  // clang-format off
#if _LIBCPP_STD_VER <= 20
  // reinterpret_cast
  {
    struct B { int b; } b;
    std::tuple<B *> t{&b};
    auto a = std::make_from_tuple<int *>(t); // expected-error-re@*:* {{static assertion failed {{.*}}Cannot constructible target type from the fields of the argument tuple.}}
    (void)a;
  }

  // const_cast
  {
    const char *str = "Hello";
    std::tuple<const char *> t{str};
    auto a = std::make_from_tuple<char *>(t); // expected-error-re@*:* {{static assertion failed {{.*}}Cannot constructible target type from the fields of the argument tuple.}}
    (void)a;
  }

  // static_cast
  {
    struct B {};
    struct C : public B {} c;
    B &br = c;
    std::tuple<const B&> t{br};
    auto a = std::make_from_tuple<const C&>(t); // expected-error-re@*:* {{static assertion failed {{.*}}Cannot constructible target type from the fields of the argument tuple.}}
    (void)a;
  }
#else
  // reinterpret_cast
  {
    struct B { int b; } b;
    std::tuple<B *> t{&b};
    auto a = std::make_from_tuple<int *>(t); // expected-error-re@*:*2 {{no matching function for call to{{.*}}}}
    (void)a;
  }

  // const_cast
  {
    const char *str = "Hello";
    std::tuple<const char *> t{str};
    auto a = std::make_from_tuple<char *>(t); // expected-error-re@*:*2 {{no matching function for call to{{.*}}}}
    (void)a;
  }

  // static_cast
  {
    struct B {};
    struct C : public B {} c;
    B &br = c;
    std::tuple<const B &> t{br};
    auto a = std::make_from_tuple<const C&>(t); // expected-error-re@*:*2 {{no matching function for call to{{.*}}}}
    (void)a;
  }
#endif
  // clang-format on
  return 0;
}
