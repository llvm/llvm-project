// RUN: %clang_cc1 -fsyntax-only -verify -Wimplicit-int-enum-cast %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify=good -Wno-implicit-enum-enum-cast %s
// RUN: %clang_cc1 -fsyntax-only -verify=good -Wc++-compat -Wno-implicit-enum-enum-cast -Wno-implicit-int-enum-cast %s
// good-no-diagnostics

enum E1 {
  E1_Zero,
  E1_One
};

enum E2 {
  E2_Zero
};

struct S {
  enum E1 e;
} s = { 12 }; // expected-warning {{implicit conversion from 'int' to enumeration type 'enum E1' is invalid in C++}} \
                 cxx-error {{cannot initialize a member subobject of type 'enum E1' with an rvalue of type 'int'}}

enum E1 foo(void) {
  int x;
  enum E1 e = 12; // expected-warning {{implicit conversion from 'int' to enumeration type 'enum E1' is invalid in C++}} \
                     cxx-error {{cannot initialize a variable of type 'enum E1' with an rvalue of type 'int'}}

  // Enum to integer is fine.
  x = e;

  // Integer to enum is not fine.
  e = x;    // expected-warning {{implicit conversion from 'int' to enumeration type 'enum E1' is invalid in C++}} \
               cxx-error {{assigning to 'enum E1' from incompatible type 'int'}}
  return x; // expected-warning {{implicit conversion from 'int' to enumeration type 'enum E1' is invalid in C++}} \
               cxx-error {{cannot initialize return object of type 'enum E1' with an lvalue of type 'int'}}
}

// Returning with the correct types is fine.
enum E1 bar(void) {
  return E1_Zero;
}

// Enum to different-enum conversion is also a C++ incompatibility, but is
// handled via a more general diagnostic, -Wimplicit-enum-enum-cast, which is
// on by default.
enum E1 quux(void) {
  enum E1 e1 = E2_Zero; // expected-warning {{implicit conversion from enumeration type 'enum E2' to different enumeration type 'enum E1'}} \
                           cxx-error {{cannot initialize a variable of type 'enum E1' with an rvalue of type 'E2'}}
  e1 = E2_Zero;         // expected-warning {{implicit conversion from enumeration type 'enum E2' to different enumeration type 'enum E1'}}   \
                           cxx-error {{assigning to 'enum E1' from incompatible type 'E2'}}
  return E2_Zero;       // expected-warning {{implicit conversion from enumeration type 'enum E2' to different enumeration type 'enum E1'}} \
                           cxx-error {{cannot initialize return object of type 'enum E1' with an rvalue of type 'E2'}}
}
