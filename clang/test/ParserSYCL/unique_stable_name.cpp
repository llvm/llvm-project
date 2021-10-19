// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -Wno-unused %s

namespace NS {
using good = double;
}

void f(int var) {
  // expected-error@+1{{expected '(' after '__builtin_sycl_unique_stable_name'}}
  __builtin_sycl_unique_stable_name int; // Correct usage is __builtin_sycl_unique_stable_name(int);

  // expected-error@+1{{expected '(' after '__builtin_sycl_unique_stable_name'}}
  __builtin_sycl_unique_stable_name{int}; // Correct usage is __builtin_sycl_unique_stable_name(int);

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  __builtin_sycl_unique_stable_name(int; // Missing paren before semicolon

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  __builtin_sycl_unique_stable_name(int, float); // Missing paren before comma

  // expected-error@+1{{unknown type name 'var'}}
  __builtin_sycl_unique_stable_name(var);
  __builtin_sycl_unique_stable_name(NS::good);

  // expected-error@+1{{expected a type}}
  __builtin_sycl_unique_stable_name(for (int i = 0; i < 10; ++i) {})
  __builtin_sycl_unique_stable_name({
    (for (int i = 0; i < 10; ++i){})})
}

template <typename T>
void f2() {
  __builtin_sycl_unique_stable_name(typename T::good_type);
}

struct S {
  class good_type {};
};

void use() {
  f2<S>();
}
