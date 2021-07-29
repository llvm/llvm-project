// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -fsyntax-only \
// RUN:   -target-cpu pwr8 -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -fsyntax-only \
// RUN:   -target-cpu pwr8 -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fsyntax-only \
// RUN:   -target-cpu pwr8 -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fsyntax-only \
// RUN:   -target-cpu pwr8 -Wall -Werror -verify %s

#include <altivec.h>
vector unsigned char test_ldrmb(char *ptr) {
  return __vec_ldrmb(ptr, 17); // expected-error {{argument value 17 is outside the valid range [1, 16]}}
}

void test_strmb(char *ptr, vector unsigned char data) {
  __vec_strmb(ptr, 17, data); // expected-error {{argument value 17 is outside the valid range [1, 16]}}
}

vector unsigned char test_ldrmbb(char *ptr) {
  return __builtin_vsx_ldrmb(ptr, 17); // expected-error {{argument value 17 is outside the valid range [1, 16]}}
}

void test_strmbb(char *ptr, vector unsigned char data) {
  __builtin_vsx_strmb(ptr, 17, data); // expected-error {{argument value 17 is outside the valid range [1, 16]}}
}
