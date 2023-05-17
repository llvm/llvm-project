// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s

template <typename... T>
void foo(T &&...Params) {
  foo<T>(Unknown); // expected-error {{expression contains unexpanded parameter pack 'T'}}\
                      expected-error {{use of undeclared identifier 'Unknown'}}
  ((foo<T>(Unknown)), ...); // expected-error {{use of undeclared identifier 'Unknown'}}
}

template <typename... U> struct A {
  template <typename... T> void foo(T &&...Params) {
    foo<T>((... + static_cast<U>(1))); // expected-error {{expression contains unexpanded parameter pack 'T'}}
  }
};
