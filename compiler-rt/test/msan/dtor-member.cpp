// RUN: %clangxx_msan %s -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t &&  %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan %s -fsanitize=memory -fno-sanitize-memory-use-after-dtor -o %t &&  %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-NO-FLAG < %t.out

// RUN: %clangxx_msan -fsanitize=memory -fsanitize-memory-use-after-dtor %s -o %t && MSAN_OPTIONS=poison_in_dtor=0 %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-NO-FLAG < %t.out

#include <sanitizer/msan_interface.h>
#include <assert.h>
#include <stdio.h>
#include <new>

struct Simple {
  int x_;
  Simple() {
    x_ = 5;
  }
  ~Simple() { }
};

int main() {
  unsigned long buf[1];
  assert(sizeof(Simple) <= sizeof(buf));

  // The placement new operator forces the object to be constructed in the
  // memory location &buf. Since objects made in this way must be explicitly
  // destroyed, there are no implicit calls inserted that would interfere with
  // test behavior.
  Simple *s = new(&buf) Simple();
  s->~Simple();

  if (__msan_test_shadow(s, sizeof(*s)) != -1)
    printf("s is poisoned\n");
  else
    printf("s is not poisoned\n");
  // CHECK: s is poisoned
  // CHECK-NO-FLAG: s is not poisoned

  return 0;
}
