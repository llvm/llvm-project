// RUN: %clangxx_asan -fsanitize-address-use-after-return=never -O %s -o %t && %run %t

#include "defines.h"
#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <stdio.h>

ATTRIBUTE_NOINLINE
void Throw() {
  int local;
  fprintf(stderr, "Throw:  %p\n", &local);
  throw 1;
}

ATTRIBUTE_NOINLINE
void ThrowAndCatch() {
  int local;
  try {
    Throw();
  } catch(...) {
    fprintf(stderr, "Catch:  %p\n", &local);
  }
}

ATTRIBUTE_NOINLINE
void TestThrow() {
  char x[32];
  fprintf(stderr, "Before: %p poisoned: %d\n", &x,
          __asan_address_is_poisoned(x + 32));
  assert(__asan_address_is_poisoned(x + 32));
  ThrowAndCatch();
  fprintf(stderr, "After:  %p poisoned: %d\n",  &x,
          __asan_address_is_poisoned(x + 32));
  assert(!__asan_address_is_poisoned(x + 32));
}

ATTRIBUTE_NOINLINE
void TestThrowInline() {
  char x[32];
  fprintf(stderr, "Before: %p poisoned: %d\n", &x,
          __asan_address_is_poisoned(x + 32));
  assert(__asan_address_is_poisoned(x + 32));
  try {
    Throw();
  } catch(...) {
    fprintf(stderr, "Catch\n");
  }
  fprintf(stderr, "After:  %p poisoned: %d\n",  &x,
          __asan_address_is_poisoned(x + 32));
  assert(!__asan_address_is_poisoned(x + 32));
}

int main(int argc, char **argv) {
  TestThrowInline();
  TestThrow();
}
