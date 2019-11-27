// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated-copy -verify
// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated-copy-dtor -DDEPRECATED_COPY_DTOR -verify
// RUN: %clang_cc1 -std=c++11 %s -Wextra -verify

#ifdef DEPRECATED_COPY_DTOR
struct A {
  int *ptr;
  ~A() { delete ptr; } // expected-warning {{definition of implicit copy constructor for 'A' is deprecated because it has a user-declared destructor}}
};

void foo() {
  A a{};
  A b = a; // expected-note {{implicit copy constructor for 'A' first required here}}
}
#else
struct B {
  B &operator=(const B &); // expected-warning {{definition of implicit copy constructor for 'B' is deprecated because it has a user-declared copy assignment operator}}
};

void bar() {
  B b1, b2(b1); // expected-note {{implicit copy constructor for 'B' first required here}}
}
#endif
