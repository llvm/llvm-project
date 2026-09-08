// Check that a report's stack traces are complete: the access, the free and the
// malloc traces must each name the whole call chain, not just the innermost
// frame or two. An unwinder that walks a frame layout the target does not use
// stops early and still prints a plausible looking report.

// RUN: %clang_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: stable-runtime

// On Alpha the fast unwinder walks a frame layout the target does not use, so
// every trace but the access one stops early. Fixed by
// https://github.com/llvm/llvm-project/pull/220231.
// XFAIL: alpha-target-arch

#include <stdlib.h>

char *p;

__attribute__((noinline)) void alloc_3(void) { p = (char *)malloc(10); }
__attribute__((noinline)) void alloc_2(void) { alloc_3(); }
__attribute__((noinline)) void alloc_1(void) { alloc_2(); }

__attribute__((noinline)) void free_3(void) { free(p); }
__attribute__((noinline)) void free_2(void) { free_3(); }
__attribute__((noinline)) void free_1(void) { free_2(); }

__attribute__((noinline)) char read_3(void) { return p[5]; }
__attribute__((noinline)) char read_2(void) { return read_3(); }
__attribute__((noinline)) char read_1(void) { return read_2(); }

int main() {
  alloc_1();
  free_1();
  return read_1();
}

// The free and malloc traces start inside the interceptor, whose frame count
// varies, so only the chain below it is pinned. main's three call sites are on
// three lines, so each trace has to name its own.

// CHECK: ERROR: AddressSanitizer: heap-use-after-free on address
// CHECK: READ of size 1 at 0x{{.*}} thread T0
// CHECK-NEXT: {{ *#0 0x.* in read_3 .*complete_stack_trace.c}}
// CHECK-NEXT: {{ *#1 0x.* in read_2 .*complete_stack_trace.c}}
// CHECK-NEXT: {{ *#2 0x.* in read_1 .*complete_stack_trace.c}}
// CHECK-NEXT: {{ *#3 0x.* in main .*complete_stack_trace.c:}}[[@LINE-12]]

// CHECK: freed by thread T0 here:
// CHECK: {{ *#[0-9]+ 0x.* in free_3 .*complete_stack_trace.c}}
// CHECK-NEXT: {{ *#[0-9]+ 0x.* in free_2 .*complete_stack_trace.c}}
// CHECK-NEXT: {{ *#[0-9]+ 0x.* in free_1 .*complete_stack_trace.c}}
// CHECK-NEXT: {{ *#[0-9]+ 0x.* in main .*complete_stack_trace.c:}}[[@LINE-19]]

// CHECK: previously allocated by thread T0 here:
// CHECK: {{ *#[0-9]+ 0x.* in alloc_3 .*complete_stack_trace.c}}
// CHECK-NEXT: {{ *#[0-9]+ 0x.* in alloc_2 .*complete_stack_trace.c}}
// CHECK-NEXT: {{ *#[0-9]+ 0x.* in alloc_1 .*complete_stack_trace.c}}
// CHECK-NEXT: {{ *#[0-9]+ 0x.* in main .*complete_stack_trace.c:}}[[@LINE-26]]
