// RUN: %clang_cc1 -fsyntax-only -Wundefined-func-template -Wundefined-var-template -verify %s

// Test that a diagnostic is emitted when an entity marked with the
// exclude_from_explicit_instantiation attribute is not defined in
// the current TU but it is used (and it is hence implicitly
// instantiated).

#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct Foo {
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void non_static_member_function(); // expected-note{{forward declaration of template entity is here}}
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static void static_member_function(); // expected-note{{forward declaration of template entity is here}}
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static int static_data_member; // expected-note{{forward declaration of template entity is here}}
  struct EXCLUDE_FROM_EXPLICIT_INSTANTIATION nested {
    static int static_member_function(); // expected-note{{forward declaration of template entity is here}}
  };
};

extern template struct Foo<int>;

void use() {
  Foo<int> foo;

  foo.non_static_member_function(); // expected-warning{{instantiation of function 'Foo<int>::non_static_member_function' required here, but no definition is available}}
  // expected-note@-1 {{add an explicit instantiation}}

  Foo<int>::static_member_function(); // expected-warning{{instantiation of function 'Foo<int>::static_member_function' required here, but no definition is available}}
  // expected-note@-1 {{add an explicit instantiation}}

  (void)Foo<int>::static_data_member; // expected-warning{{instantiation of variable 'Foo<int>::static_data_member' required here, but no definition is available}}
  // expected-note@-1 {{add an explicit instantiation}}

  Foo<int>::nested::static_member_function(); // expected-warning{{instantiation of function 'Foo<int>::nested::static_member_function' required here, but no definition is available}}
  // expected-note@-1 {{add an explicit instantiation}}
}
