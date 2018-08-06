// Check to make sure that we are actually filtering records from the basic mode
// logging implementation.

// RUN: %clangxx_xray -std=c++11 %s -o %t -g
// RUN: rm -f basic-filtering-*
// RUN: XRAY_OPTIONS="patch_premain=true xray_mode=xray-basic verbosity=1 \
// RUN:     xray_logfile_base=basic-filtering- \
// RUN:     xray_naive_log_func_duration_threshold_us=1000 \
// RUN:     xray_naive_log_max_stack_depth=2" %run %t 2>&1 | \
// RUN:     FileCheck %s
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t \
// RUN:     "`ls basic-filtering-* | head -1`" | \
// RUN:     FileCheck %s --check-prefix TRACE
// RUN: rm -f basic-filtering-*
//
// Now check support for the XRAY_BASIC_OPTIONS environment variable.
// RUN: XRAY_OPTIONS="patch_premain=true xray_mode=xray-basic verbosity=1 \
// RUN:     xray_logfile_base=basic-filtering-" \
// RUN: XRAY_BASIC_OPTIONS="func_duration_threshold_us=1000 max_stack_depth=2" \
// RUN:     %run %t 2>&1 | FileCheck %s
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t \
// RUN:     "`ls basic-filtering-* | head -1`" | \
// RUN:     FileCheck %s --check-prefix TRACE
// RUN: rm -f basic-filtering-*
//
// REQUIRES: x86_64-target-arch
// REQUIRES: built-in-llvm-tree

#include <cstdio>
#include <time.h>

[[clang::xray_always_instrument]] void __attribute__((noinline)) filtered() {
  printf("filtered was called.\n");
}

[[clang::xray_always_instrument]] void __attribute__((noinline)) beyond_stack() {
  printf("beyond stack was called.\n");
}

[[clang::xray_always_instrument]] void __attribute__((noinline))
always_shows() {
  struct timespec sleep;
  sleep.tv_nsec = 2000000;
  sleep.tv_sec = 0;
  struct timespec rem;
  while (nanosleep(&sleep, &rem) == -1)
    sleep = rem;
  printf("always_shows was called.\n");
  beyond_stack();
}

[[clang::xray_always_instrument]] int main(int argc, char *argv[]) {
  filtered();     // CHECK: filtered was called.
  always_shows(); // CHECK: always_shows was called.
  // CHECK: beyond stack was called.
}

// TRACE-NOT: - { type: 0, func-id: {{.*}}, function: {{.*filtered.*}}, {{.*}} }
// TRACE-NOT: - { type: 0, func-id: {{.*}}, function: {{.*beyond_stack.*}}, {{.*}} }
// TRACE-DAG: - { type: 0, func-id: [[FID:[0-9]+]], function: {{.*always_shows.*}}, cpu: {{.*}}, thread: {{.*}}, kind: function-enter, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID]], function: {{.*always_shows.*}}, cpu: {{.*}}, thread: {{.*}}, kind: function-{{exit|tail-exit}}, tsc: {{[0-9]+}} }
