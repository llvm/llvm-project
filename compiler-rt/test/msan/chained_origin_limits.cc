// This test program creates a very large number of unique histories.

// Heap origin.
// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O3 %s -o %t

// RUN: MSAN_OPTIONS=origin_history_size=7 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK7 < %t.out

// RUN: MSAN_OPTIONS=origin_history_size=2 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK2 < %t.out

// RUN: MSAN_OPTIONS=origin_history_per_stack_limit=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-PER-STACK < %t.out

// RUN: MSAN_OPTIONS=origin_history_size=7,origin_history_per_stack_limit=0 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK7 < %t.out

// Stack origin.
// RUN: %clangxx_msan -DSTACK -fsanitize-memory-track-origins=2 -O3 %s -o %t

// RUN: MSAN_OPTIONS=origin_history_size=7 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK7 < %t.out

// RUN: MSAN_OPTIONS=origin_history_size=2 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK2 < %t.out

// RUN: MSAN_OPTIONS=origin_history_per_stack_limit=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-PER-STACK < %t.out

// RUN: MSAN_OPTIONS=origin_history_size=7,origin_history_per_stack_limit=0 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK7 < %t.out


// Heap origin, with calls.
// RUN: %clangxx_msan -mllvm -msan-instrumentation-with-call-threshold=0 -fsanitize-memory-track-origins=2 -O3 %s -o %t

// RUN: MSAN_OPTIONS=origin_history_size=7 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK7 < %t.out

// RUN: MSAN_OPTIONS=origin_history_size=2 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK2 < %t.out

// RUN: MSAN_OPTIONS=origin_history_per_stack_limit=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-PER-STACK < %t.out

// RUN: MSAN_OPTIONS=origin_history_size=7,origin_history_per_stack_limit=0 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK7 < %t.out


// Stack origin, with calls.
// RUN: %clangxx_msan -DSTACK -mllvm -msan-instrumentation-with-call-threshold=0 -fsanitize-memory-track-origins=2 -O3 %s -o %t

// RUN: MSAN_OPTIONS=origin_history_size=7 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK7 < %t.out

// RUN: MSAN_OPTIONS=origin_history_size=2 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK2 < %t.out

// RUN: MSAN_OPTIONS=origin_history_per_stack_limit=1 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-PER-STACK < %t.out

// RUN: MSAN_OPTIONS=origin_history_size=7,origin_history_per_stack_limit=0 not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK7 < %t.out

// XFAIL: target-is-mips64el

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char *buf, *cur, *end;
void init() {
  buf = new char[1000];
#ifdef STACK
  char stackbuf[1000];
  char *volatile p = stackbuf;
  memcpy(buf, p, 1000);
#endif
  cur = buf;
  end = buf + 1000;
}

void line_flush() {
  char *p;
  for (p = cur - 1; p >= buf; --p)
    if (*p == '\n')
      break;
  if (p >= buf) {
    size_t write_sz = p - buf + 1;
    // write(2, buf, write_sz);
    memmove(buf, p + 1, end - p - 1);
    cur -= write_sz;
  }
}

void buffered_write(const char *p, size_t sz) {
  while (sz > 0) {
    size_t copy_sz = end - cur;
    if (sz < copy_sz) copy_sz = sz;
    memcpy(cur, p, copy_sz);
    cur += copy_sz;
    sz -= copy_sz;
    line_flush();
  }
}

void fn1() {
  buffered_write("a\n", 2);
}

void fn2() {
  buffered_write("a\n", 2);
}

void fn3() {
  buffered_write("a\n", 2);
}

int main(void) {
  init();
  for (int i = 0; i < 2000; ++i) {
    fn1();
    fn2();
    fn3();
  }
  return buf[50];
}

// CHECK7: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK7-NOT: Uninitialized value was stored to memory at
// CHECK7: Uninitialized value was stored to memory at
// CHECK7-NOT: Uninitialized value was stored to memory at
// CHECK7: Uninitialized value was stored to memory at
// CHECK7-NOT: Uninitialized value was stored to memory at
// CHECK7: Uninitialized value was stored to memory at
// CHECK7-NOT: Uninitialized value was stored to memory at
// CHECK7: Uninitialized value was stored to memory at
// CHECK7-NOT: Uninitialized value was stored to memory at
// CHECK7: Uninitialized value was stored to memory at
// CHECK7-NOT: Uninitialized value was stored to memory at
// CHECK7: Uninitialized value was stored to memory at
// CHECK7-NOT: Uninitialized value was stored to memory at
// CHECK7: Uninitialized value was created

// CHECK2: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK2-NOT: Uninitialized value was stored to memory at
// CHECK2: Uninitialized value was stored to memory at
// CHECK2-NOT: Uninitialized value was stored to memory at
// CHECK2: Uninitialized value was created

// CHECK-PER-STACK: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK-PER-STACK: Uninitialized value was stored to memory at
// CHECK-PER-STACK: in fn3
// CHECK-PER-STACK: Uninitialized value was stored to memory at
// CHECK-PER-STACK: in fn2
// CHECK-PER-STACK: Uninitialized value was stored to memory at
// CHECK-PER-STACK: in fn1
// CHECK-PER-STACK: Uninitialized value was created

// CHECK-UNLIMITED: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was stored to memory at
// CHECK-UNLIMITED: Uninitialized value was created
