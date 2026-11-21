// RUN: %clang_cc1 -fsyntax-only -std=c++2c -ferror-limit 1 -verify %s

// GH214917: don't crash expanding an expansion statement after a fatal error.

unknown_type a; // expected-error {{unknown type name 'unknown_type'}}
unknown_type b;
// expected-error@* {{too many errors emitted}}

struct T {
  int x, y;
};

void f() {
  template for (auto i : T{42, 10}) {
    (void)i;
  }
}

struct S {
  Foo(Foo &&);
};

template <int> std { struct tuple_size; };

struct V {
  int i, j;
};

template <> struct std::tuple_size<V> {
  static const int value = 2;
};
template <int I> struct std::tuple_element<I, V> {
  using type = int;
};

template <> struct std::tuple_size<V> {
  static const int value = 2;
};
template <int I> struct std::tuple_element<I, V> {
  using type = int;
};

void bar() { template for (auto i : V{42, 10}) i += (V.i == 42); }
