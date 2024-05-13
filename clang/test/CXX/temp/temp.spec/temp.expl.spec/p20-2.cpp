// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
void f(T);

template<typename T>
struct A {
  // expected-error@+1{{cannot declare an explicit specialization in a friend}}
  template <> friend void f<>(int) {}
};

// Makes sure implicit instantiation here does not trigger
// the assertion "Member specialization must be an explicit specialization"
void foo(void) {
    A<int> a;
}
