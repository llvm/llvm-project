// Tests that invalid base-specifiers no longer crash the compiler.
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

namespace GH147186 {

class X; // expected-note {{forward declaration of 'GH147186::X'}} expected-note {{forward declaration of 'GH147186::X'}}

class A : X { // expected-error {{base class has incomplete type}}
};

class Y : int { // expected-error {{expected class name}}
};

class Z : X*, virtual int { // expected-error {{base class has incomplete type}} expected-error {{expected class name}}
};
}
