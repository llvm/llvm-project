// RUN: %clang_cc1 -triple x86_64-apple-darwin -fcxx-exceptions -verify -std=c++2a -fmodules -I%S/Inputs %s -fno-modules-error-recovery

#pragma clang module build compare
module compare {
  explicit module cmp {}
  explicit module other {}
}
#pragma clang module contents
#pragma clang module begin compare.cmp
#include "std-compare.h"
#pragma clang module end
#pragma clang module endbuild

struct CC { CC(...); };

void a() { void(0 <=> 0); } // expected-error {{include <compare>}}

struct A {
  CC operator<=>(const A&) const = default; // expected-error {{include <compare>}}
};
auto va = A() <=> A(); // expected-note {{required here}}

#pragma clang module import compare.other

// expected-note@std-compare.h:* 2+{{previous definition}}

void b() { void(0 <=> 0); } // expected-error 1+{{definition of 'strong_ordering' must be imported}}

struct B {
  CC operator<=>(const B&) const = default; // expected-error 1+{{definition of 'strong_ordering' must be imported}}
};
auto vb = B() <=> B(); // expected-note {{required here}}

#pragma clang module import compare.cmp

void c() { void(0 <=> 0); }

struct C {
  CC operator<=>(const C&) const = default;
};
auto vc = C() <=> C();


#pragma clang module build compare2
module compare2 {}
#pragma clang module contents
#pragma clang module begin compare2
#include "std-compare.h"
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import compare2

void g() { void(0.0 <=> 0.0); }
