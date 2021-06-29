// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// 
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>

int main(int argc, char *argv[]) {
  uint64_t a = 10;
  dfsan_set_label(1, &a, sizeof(a));

  // Manually compute origin address for &a.
  // See x86 MEM_TO_ORIGIN macro for logic to replicate here.
  // Alignment is also needed after to MEM_TO_ORIGIN.
  uint64_t origin_addr =
      (((uint64_t)&a ^ 0x500000000000ULL) + 0x100000000000ULL) & ~0x3ULL;

  // Take the address we computed, and store 0 in it to mess it up.
  asm("mov %0, %%rax": :"r"(origin_addr));
  asm("movq $0, (%rax)");
  dfsan_print_origin_trace(&a, "invalid");
}

// CHECK: Taint value 0x1 (at {{.*}}) origin tracking (invalid)
// CHECK: Taint value 0x1 (at {{.*}}) has invalid origin tracking. This can be a DFSan bug.
