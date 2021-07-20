// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-cpu pwr9 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s

extern unsigned int ui;

void test_builtin_ppc_cmprb() {
  int res = __builtin_ppc_cmprb(3, ui, ui); // expected-error {{argument value 3 is outside the valid range [0, 1]}}
}
