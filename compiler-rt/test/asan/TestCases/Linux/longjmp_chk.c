// Verify that use of longjmp() in a _FORTIFY_SOURCE'd library (without ASAN)
// is correctly intercepted such that the stack is unpoisoned.
// Note: it is essential that the external library is not built with ASAN,
// otherwise it would be able to unpoison the stack before use.
//
// RUN: %clang -DIS_LIBRARY -D_FORTIFY_SOURCE=2 -O2 %s -c -o %t.o
// RUN: %clang_asan -O2 %s %t.o -o %t
// RUN: %run %t

#ifdef IS_LIBRARY
/* the library */
#include <setjmp.h>
#include <assert.h>
#include <sanitizer/asan_interface.h>

static jmp_buf jenv;

void external_callme(void (*callback)(void)) {
  if (setjmp(jenv) == 0) {
    callback();
  }
}

void external_longjmp(char *msg) {
  longjmp(jenv, 1);
}

void external_check_stack(void) {
  char buf[256] = "";
  for (int i = 0; i < 256; i++) {
    assert(!__asan_address_is_poisoned(buf + i));
  }
}
#else
/* main program */
extern void external_callme(void (*callback)(void));
extern void external_longjmp(char *msg);
extern void external_check_stack(void);

static void callback(void) {
  char msg[16];   /* Note: this triggers addition of a redzone. */
  /* Note: msg is passed to prevent compiler optimization from removing it. */
  external_longjmp(msg);
}

int main() {
  external_callme(callback);
  external_check_stack();
  return 0;
}
#endif
