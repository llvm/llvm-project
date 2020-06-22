// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffine-grained-bitfield-accesses \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple riscv64-linux-gnu -ffine-grained-bitfield-accesses \
// RUN:   -emit-llvm -o - %s | FileCheck %s

// Note: This test checks the X86-64 and RISC-V targets in order to explore
// behaviour when i8/i16 are native integer widths (X86-64) and when they're
// not (RISC-V).

struct S4 {
  unsigned long f1:28;
  unsigned long f2:4;
  unsigned long f3:12;
};
struct S4 a4;

struct S5 {
  unsigned long f1:28;
  unsigned long f2:4;
  unsigned long f3:28;
  unsigned long f4:4;
  unsigned long f5:12;
};
struct S5 a5;

// CHECK: %struct.S4 = type { i32, i16 }
// CHECK-NOT: %struct.S4 = type { i48 }
// CHECK: %struct.S5 = type { i32, i32, i16, [6 x i8] }
// CHECK-NOT: %struct.S5 = type { i80 }
