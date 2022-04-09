// Based on C++20 10.2 example 6.

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -verify -o %t

export module M;

static int f();  // expected-note {{previous declaration is here}} #1
                 // error: #1 gives internal linkage
export int f();  // expected-error {{cannot export redeclaration 'f' here since the previous declaration has internal linkage}}
struct S;        // expected-note {{previous declaration is here}} #2
                 // error: #2 gives module linkage
export struct S; // expected-error {{cannot export redeclaration 'S' here since the previous declaration has module linkage}}

namespace {
namespace N {
extern int x; // expected-note {{previous declaration is here}} #3
}
} // namespace
  // error: #3 gives internal linkage
export int N::x; // expected-error {{cannot export redeclaration 'x' here since the previous declaration has internal linkage}}
                 // expected-error@-1 {{declaration of 'x' with internal linkage cannot be exported}}
