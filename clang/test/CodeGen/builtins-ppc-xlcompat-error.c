// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -fsyntax-only \
// RUN:   -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -fsyntax-only \
// RUN:   -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fsyntax-only \
// RUN:   -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fsyntax-only \
// RUN:   -Wall -Werror -verify %s

long long lla, llb;
int ia, ib;

void test_trap(void) {
  __tdw(lla, llb, 50); //expected-error {{argument value 50 is outside the valid range [0, 31]}}
  __tw(ia, ib, 50); //expected-error {{argument value 50 is outside the valid range [0, 31]}}
}
