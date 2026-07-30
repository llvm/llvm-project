// RUN: %clang_cc1 -verify %s

// FIXME: We could in principle support cases like this (particularly, cases
// where the statement-expression contains no labels).
template <typename... T> void f1() {
  int arr[] = {
    ({
      T(); // expected-error {{unexpanded parameter pack}}
    }) ... // expected-error {{does not contain any unexpanded parameter packs}}
  };
}

// FIXME: The error for this isn't ideal; it'd be preferable to say that pack
// expansion of a statement expression is not permitted.
template <typename... T> void f2() {
  [] {
    int arr[] = {
      T() + ({
      foo:
        T t; // expected-error {{unexpanded parameter pack}}
        goto foo;
        0;
      }) ...
    };
  };
}

template <typename... T> void f3() {
  ({
    int arr[] = {
      [] {
      foo:
        T t; // OK, expanded within compound statement
        goto foo;
        return 0;
      } ...
    };
  });
}
