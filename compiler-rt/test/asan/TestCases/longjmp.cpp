// RUN: %clangxx_asan -fsanitize-address-use-after-return=never -O %s -o %t && %run %t

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <sanitizer/asan_interface.h>

static jmp_buf buf;

int main() {
  char x[32];
  fprintf(stderr, "\nTestLongJmp\n");
  fprintf(stderr, "Before: %p poisoned: %d\n", &x,
          __asan_address_is_poisoned(x + 32));
  assert(__asan_address_is_poisoned(x + 32));
  if (0 == setjmp(buf))
    longjmp(buf, 1);
  fprintf(stderr, "After:  %p poisoned: %d\n",  &x,
          __asan_address_is_poisoned(x + 32));
  assert(!__asan_address_is_poisoned(x + 32));
}
