// RUN: %clang_cc1 -std=c++20 -verify %s

auto f = [] {
  int a;
  auto g = [](auto v) {
    {
      struct Nested {
        constexpr int value = v;
        // expected-error@-1 {{non-static data member cannot be constexpr; did you intend to make it static?}}
        // expected-error@-2 {{static data member 'value' not allowed in local struct 'Nested'}}
      };
      return Nested::value;
    }
  };
  g(a);
};
