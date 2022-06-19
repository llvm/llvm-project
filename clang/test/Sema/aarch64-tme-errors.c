// RUN: %clang_cc1 -triple aarch64-eabi -verify %s

#include "arm_acle.h"

void test_no_tme_funcs(void) {
  __tstart();         // expected-error{{call to undeclared function '__tstart'; ISO C99 and later do not support implicit function declarations}}
  __builtin_tstart(); // expected-error{{use of unknown builtin '__builtin_tstart'}}
}
