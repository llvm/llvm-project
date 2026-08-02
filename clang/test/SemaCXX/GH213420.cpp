// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fms-compatibility -verify \
// RUN:   -Wno-missing-declarations -Wno-unused-value %s

void foo() {
  [] {
    struct { // expected-error {{anonymous structs and classes must be class members}}
      void bar(int & = "") {} // expected-error {{non-const lvalue reference to type 'int' cannot bind}}
                              // expected-note@-1 {{passing argument to parameter here}}
                              // expected-error@-2 {{functions cannot be declared in an anonymous struct}}
    };
  };
}
