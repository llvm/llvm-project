// RUN: %clang_dfsan -mllvm -dfsan-fast-16-labels=true -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out

#include <sanitizer/dfsan_interface.h>

int main(int argc, char *argv[]) {
  uint64_t a = 10;
  dfsan_set_label(8, &a, sizeof(a));
  size_t origin_addr =
      (((size_t)&a & ~0x700000000000LL + 0x200000000000LL) & ~0x3UL);
  asm("mov %0, %%rax": :"r"(origin_addr));
  asm("movq $0, (%rax)");
  dfsan_print_origin_trace(&a, "invalid");
}

// CHECK: Taint value 0x8 (at {{.*}}) origin tracking (invalid)
// CHECK: Taint value 0x8 (at {{.*}}) has invalid origin tracking. This can be a DFSan bug.
