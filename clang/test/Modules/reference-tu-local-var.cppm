// RUN: %clang_cc1 -std=c++23 %s -verify -fsyntax-only
export module M;
static int local;
export inline int exposure1() { return local; } // expected-warning {{TU local entity 'local' is exposed}}

static int local2 = 43;
export extern "C++" {
inline int exposure2() { return local2; } // expected-warning {{TU local entity 'local2' is exposed}}
}
