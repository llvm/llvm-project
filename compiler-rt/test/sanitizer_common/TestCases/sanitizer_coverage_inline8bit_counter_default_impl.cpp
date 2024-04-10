// Tests the default implementation of callbacks for
// -fsanitize-coverage=inline-8bit-counters,pc-table

// REQUIRES: has_sancovcc,stable-runtime,linux,x86_64-target-arch

/// In glibc 2.39+, fprintf has a nonnull attribute. Disable nonnull-attribute,
/// which would increase counters for ubsan.
// RUN: %clangxx -O0 %s -fsanitize-coverage=inline-8bit-counters,pc-table -fno-sanitize=nonnull-attribute -o %t
// RUN: rm -f %t-counters %t-pcs
// RUN: env %tool_options="cov_8bit_counters_out=%t-counters cov_pcs_out=%t-pcs verbosity=1" %run %t 2>&1 | FileCheck %s

// Check the file sizes
// RUN: wc -c %t-counters | grep "^2 "
// RUN: wc -c %t-pcs | grep "^32 "

#include <stdio.h>

__attribute__((noinline)) void foo() {}
int main() {
  foo();
  foo();
  fprintf(stderr, "PASS\n");
  // CHECK: PASS
  // CHECK: cov_8bit_counters_out: written {{.*}} bytes to {{.*}}-counter
  // CHECK: cov_pcs_out: written {{.*}} bytes to {{.*}}-pcs
}
