// RUN: %clangxx_msan %s -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -fsanitize-memory-track-origins -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out

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

  return s->x_;

  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main.*use-after-dtor.cc:}}[[@LINE-3]]

  // CHECK-ORIGINS: Memory was marked as uninitialized
  // CHECK-ORIGINS: {{#0 0x.* in __sanitizer_dtor_callback}}
  // CHECK-ORIGINS: {{#1 0x.* in Simple::~Simple}}

  // CHECK: SUMMARY: MemorySanitizer: use-of-uninitialized-value {{.*main}}
}
