// RUN: %clangxx_asan -O %s -o %t && %run %t

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

void TestThrow() {
  char x[32];
  fprintf(stderr, "Before: %p poisoned: %d\n", &x,
          __asan_address_is_poisoned(x + 32));
  assert(__asan_address_is_poisoned(x + 32));
  ThrowAndCatch();
  fprintf(stderr, "After:  %p poisoned: %d\n",  &x,
          __asan_address_is_poisoned(x + 32));
  // FIXME: Invert this assertion once we fix
  // https://code.google.com/p/address-sanitizer/issues/detail?id=258
  // This assertion works only w/o UAR.
  if (!__asan_get_current_fake_stack())
    assert(!__asan_address_is_poisoned(x + 32));
}

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
  // FIXME: Invert this assertion once we fix
  // https://code.google.com/p/address-sanitizer/issues/detail?id=258
  // This assertion works only w/o UAR.
  if (!__asan_get_current_fake_stack())
    assert(!__asan_address_is_poisoned(x + 32));
}

int main(int argc, char **argv) {
  TestThrowInline();
  TestThrow();
}
