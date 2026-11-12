// RUN: %clang_cc1 %s

template <typename T>
struct Foo {
  template <typename U>
  struct Bar {};

  // The templated declaration for class Bar should not be instantiated when
  // Foo<int> is. This is to protect against PR5848; for now, this "parses" but
  // requires a rewrite of the templated friend code to be properly fixed.
  template <typename U>
  friend struct Bar;
};

Foo<int> x;
