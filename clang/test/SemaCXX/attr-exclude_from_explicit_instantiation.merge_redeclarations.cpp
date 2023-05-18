// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test that we properly merge the exclude_from_explicit_instantiation
// attribute on redeclarations.

#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct Foo {
  // Declaration without the attribute, definition with the attribute.
  void func1();

  // Declaration with the attribute, definition without the attribute.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void func2();

  // Declaration with the attribute, definition with the attribute.
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void func3();
};

template <class T>
EXCLUDE_FROM_EXPLICIT_INSTANTIATION void Foo<T>::func1() {
  using Fail = typename T::invalid; // expected-error{{no type named 'invalid' in 'Empty'}}
}

template <class T>
void Foo<T>::func2() {
  using Fail = typename T::invalid; // expected-error{{no type named 'invalid' in 'Empty'}}
}

template <class T>
EXCLUDE_FROM_EXPLICIT_INSTANTIATION void Foo<T>::func3() {
  using Fail = typename T::invalid; // expected-error{{no type named 'invalid' in 'Empty'}}
}

struct Empty { };
extern template struct Foo<Empty>;

int main() {
  Foo<Empty> foo;
  foo.func1(); // expected-note{{in instantiation of}}
  foo.func2(); // expected-note{{in instantiation of}}
  foo.func3(); // expected-note{{in instantiation of}}
}
