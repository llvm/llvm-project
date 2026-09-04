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

enum E1 comma1(void) {
  return ((void)0, E1_One);
}

enum E1 comma2(void) {
  enum E1 x;
  return
    (x = 12,  // expected-warning {{implicit conversion from 'int' to enumeration type 'enum E1' is invalid in C++}} \
                 cxx-error {{assigning to 'enum E1' from incompatible type 'int'}}
    E1_One);
}

enum E1 comma3(void) {
  enum E1 x;
  return ((void)0, foo()); // Okay, no conversion in C++
}

enum E1 comma4(void) {
  return ((void)1, 2); // expected-warning {{implicit conversion from 'int' to enumeration type 'enum E1' is invalid in C++}} \
                          cxx-error {{cannot initialize return object of type 'enum E1' with an rvalue of type 'int'}}
}

// The branches of a conditional operand are each converted to the context
// type, so a conditional between enumerators of the target type is fine in
// C++ and must not be diagnosed here either.
enum E1 comma5(int c) {
  return ((void)0, c ? E1_One : E1_Zero); // Okay, no conversion in C++
}

#ifdef __cplusplus
// In C++ the enumerators already have the enumeration type, so the conditional
// has type E1 and there is nothing to convert.
static_assert(__is_same(decltype(true ? E1_One : E1_Zero), E1), "");
#endif

enum E1 comma6(int c) {
  return ((void)0, c ? E1_One : 2); // expected-warning {{implicit conversion from 'int' to enumeration type 'enum E1' is invalid in C++}} \
                                       cxx-error {{cannot initialize return object of type 'enum E1' with an rvalue of type 'int'}}
}
