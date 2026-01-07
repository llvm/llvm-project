// RUN: %check_clang_tidy -std=c++20 %s modernize-type-traits %t

namespace std {
template <class> struct tuple_size {
  static const int value = 1;
};
template <int, class> struct tuple_element {
  using type = int;
};
}

struct A {};
template <int> int get(const A&);

auto [a] = A();
