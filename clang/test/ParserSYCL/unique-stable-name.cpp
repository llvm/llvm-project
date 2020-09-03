// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused %s

namespace NS{};

void f(int var) {
  // expected-error@+1{{expected '(' after '__builtin_unique_stable_name'}}
  __builtin_unique_stable_name int;
  // expected-error@+1{{expected '(' after '__builtin_unique_stable_name'}}
  __builtin_unique_stable_name {int};

  __builtin_unique_stable_name(var);
  // expected-error@+1{{use of undeclared identifier 'bad_var'}}
  __builtin_unique_stable_name(bad_var);
  // expected-error@+1{{use of undeclared identifier 'bad'}}
  __builtin_unique_stable_name(bad::type);
  // expected-error@+1{{no member named 'still_bad' in namespace 'NS'}}
  __builtin_unique_stable_name(NS::still_bad);
}

template <typename T>
void f2() {
  // expected-error@+1{{no member named 'bad_val' in 'S'}}
  __builtin_unique_stable_name(T::bad_val);
  // expected-error@+1{{no type named 'bad_type' in 'S'}}
  __builtin_unique_stable_name(typename T::bad_type);
}

struct S{};

void use() {
  // expected-note@+1{{in instantiation of}}
  f2<S>();
}
