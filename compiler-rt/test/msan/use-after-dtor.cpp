// RUN: %clangxx_msan %s -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-UAD < %t.out

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-UAD < %t.out

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-UAD < %t.out

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -fsanitize-memory-track-origins -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefixes=CHECK-UAD,CHECK-ORIGINS < %t.out

// RUN: %clangxx_msan %s -fno-sanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t > %t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-UAD-OFF < %t.out

#include <sanitizer/msan_interface.h>
#include <assert.h>
#include <stdio.h>
#include <new>

struct Simple {
  int x_;
  Simple() {
    x_ = 5;
  }
  ~Simple() {
    x_ += 1;
  }
};

int main() {
  unsigned long buf[1];
  assert(sizeof(Simple) <= sizeof(buf));

  Simple *s = new(&buf) Simple();
  s->~Simple();

  fprintf(stderr, "\n");  // Need output to parse for CHECK-UAD-OFF case
  return s->x_;

  // CHECK-UAD: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK-UAD: {{#0 0x.* in main.*use-after-dtor.cpp:}}[[@LINE-3]]

  // CHECK-ORIGINS: Memory was marked as uninitialized
  // CHECK-ORIGINS: {{#0 0x.* in __sanitizer_dtor_callback}}
  // CHECK-ORIGINS: {{#1 0x.* in .*~Simple}}

  // CHECK-UAD: SUMMARY: MemorySanitizer: use-of-uninitialized-value {{.*main}}
  // CHECK-UAD-OFF-NOT: SUMMARY: MemorySanitizer: use-of-uninitialized-value
}
