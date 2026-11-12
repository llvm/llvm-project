// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

// Simple form
int ar1[10];

// Element type cannot be:
// - (cv) void
volatile void ar2[10]; // expected-error {{incomplete element type 'volatile void'}}
// - a reference
int& ar3[10]; // expected-error {{array of references}}
// - a function type
typedef void Fn();
Fn ar4[10]; // expected-error {{array of functions}}
// - an abstract class
struct Abstract { virtual void fn() = 0; }; // expected-note {{pure virtual}}
Abstract ar5[10]; // expected-error {{abstract class}}

// If we have a size, it must be greater than zero.
int ar6[-1]; // expected-error {{array with a negative size}}
int ar7[0u]; // expected-warning {{zero size arrays are an extension}}

// An array with unknown bound is incomplete.
int ar8[]; // expected-error {{needs an explicit size or an initializer}}
// So is an array with an incomplete element type.
struct Incomplete; // expected-note {{forward declaration}}
Incomplete ar9[10]; // expected-error {{incomplete type}}
// Neither of which should be a problem in situations where no complete type
// is required. (PR5048)
void fun(int p1[], Incomplete p2[10]);
extern int ear1[];
extern Incomplete ear2[10];

// cv migrates to element type
typedef const int cint;
extern cint car1[10];
typedef int intar[10];
// thus this is a valid redeclaration
extern const intar car1;

// Check that instantiation works properly when the element type is a template.
template <typename T> struct S {
  typename T::type x; // expected-error {{has no members}}
};
S<int> ar10[10]; // expected-note {{requested here}}

// Ensure that negative array size errors include the name of the declared
// array as this is often used to simulate static_assert with template
// instantiations, placing the 'error message' in the declarator name.
int
user_error_message
[-1]; // expected-error {{user_error_message}}
typedef int
another_user_error_message
[-1]; // expected-error {{another_user_error_message}}
