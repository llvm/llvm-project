// RUN: %clang_cc1 -triple aarch64-eabi -verify %s

#include "arm_acle.h"

void test_no_tme_funcs(void) {
  __tstart();         // expected-warning{{implicit declaration of function '__tstart'}}
  __builtin_tstart(); // expected-error{{use of unknown builtin '__builtin_tstart'}}
}
