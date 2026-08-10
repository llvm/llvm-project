// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated -verify
// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated -verify -fms-compatibility
// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated-copy-with-user-provided-dtor -verify
// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated-copy-with-user-provided-dtor -verify -fms-compatibility

struct A {
  ~A(); // expected-warning {{definition of implicit copy constructor for 'A' is deprecated because it has a user-provided destructor}}
};

void test() {
  A a1;
  A a2(a1); // expected-note {{implicit copy constructor for 'A' first required here}}
}
