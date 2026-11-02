#ifndef TEST__WRAPPERS_H
#define TEST__WRAPPERS_H

#include "make_signed.h"

int my_wrapper(unsigned a) {
  // direct wrapper
  return my_make_signed(a);
}

int my_wrapper_2(unsigned a) {
  volatile int test = a;
  (void)test;
  return my_make_signed(a);
}

#endif
