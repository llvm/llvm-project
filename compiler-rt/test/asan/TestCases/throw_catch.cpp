// RUN: %clangxx_asan -fsanitize-address-use-after-return=never -O %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>
#include <sanitizer/asan_interface.h>

__attribute__((noinline))
void Throw() {
  int local;
  fprintf(stderr, "Throw:  %p\n", &local);
  throw 1;
}

__attribute__((noinline))
void ThrowAndCatch() {
  int local;
  try {
    Throw();
  } catch(...) {
    fprintf(stderr, "Catch:  %p\n", &local);
  }
}

__attribute__((noinline))
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

__attribute__((noinline))
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
