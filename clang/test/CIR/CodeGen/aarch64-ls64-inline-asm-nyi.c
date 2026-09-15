// RUN: not %clang_cc1 -triple aarch64 -target-feature +ls64 -fclangir \
// RUN:   -emit-cir %s -o /dev/null 2>&1 | FileCheck %s

struct data512 {
  unsigned long long data[8];
};

void store(const struct data512 *input, void *addr) {
  __asm__ volatile("st64b %0, [%1]"
                   :
                   : "r"(*input), "r"(addr)
                   : "memory");
}

// CHECK: error: ClangIR code gen Not Yet Implemented: AArch64 LS64
// CHECK-SAME: scalarizable asm operand
