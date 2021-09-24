// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu pwr10 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu pwr10 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-cpu pwr10 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-cpu pwr10 \
// RUN:   -fsyntax-only -Wall -Werror -verify %s

#include <altivec.h>

vector unsigned char vuca;
vector unsigned short vusa;
vector unsigned int vuia;
vector unsigned long long vulla;

unsigned long long test_vec_cntm_uc(void) {
  return vec_cntm(vuca, -1); // expected-error 1+ {{argument value 255 is outside the valid range [0, 1]}}
}

unsigned long long test_vec_cntm_us(void) {
  return vec_cntm(vusa, -1); // expected-error 1+ {{argument value 255 is outside the valid range [0, 1]}}
}

unsigned long long test_vec_cntm_ui(void) {
  return vec_cntm(vuia, 2); // expected-error 1+ {{argument value 2 is outside the valid range [0, 1]}}
}

unsigned long long test_vec_cntm_ull(void) {
  return vec_cntm(vulla, 2); // expected-error 1+ {{argument value 2 is outside the valid range [0, 1]}}
}
