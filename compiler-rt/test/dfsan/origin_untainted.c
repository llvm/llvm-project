// RUN: %clang_dfsan -gmlt -mllvm -dfsan-fast-16-labels=true -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out

#include <sanitizer/dfsan_interface.h>

int main(int argc, char *argv[]) {
  uint64_t a = 10;
  dfsan_print_origin_trace(&a, NULL);
}

// CHECK: DFSan: no tainted value at {{.*}}
