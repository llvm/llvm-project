// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// 
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>

int main(int argc, char *argv[]) {
  uint64_t a = 10;
  dfsan_set_label(1, &a, sizeof(a));
  size_t origin_addr =
      (((size_t)&a & ~0x600000000000LL + 0x100000000000LL) & ~0x3UL);
  asm("mov %0, %%rax": :"r"(origin_addr));
  asm("movq $0, (%rax)");
  dfsan_print_origin_trace(&a, "invalid");
}

// CHECK: Taint value 0x1 (at {{.*}}) origin tracking (invalid)
// CHECK: Taint value 0x1 (at {{.*}}) has invalid origin tracking. This can be a DFSan bug.
