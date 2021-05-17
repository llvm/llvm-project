// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated -verify
// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated-copy-dtor -verify
// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated-copy-with-dtor -verify

class A {
public:
   ~A() = default; // expected-warning {{definition of implicit copy constructor for 'A' is deprecated because it has a user-declared destructor}}
};

void test() {
  A a1;
  A a2 = a1; // expected-note {{in implicit copy constructor for 'A' first required here}}
}
