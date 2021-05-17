// RUN: %clang_dfsan -gmlt -mllvm -dfsan-fast-16-labels=true %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>

int main(int argc, char *argv[]) {
  uint64_t a = 10;
  dfsan_set_label(8, &a, sizeof(a));
  dfsan_print_origin_trace(&a, NULL);
}

// CHECK: DFSan: origin tracking is not enabled. Did you specify the -dfsan-track-origins=1 option?
