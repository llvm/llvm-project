// RUN: %clang_cc1 -pg -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -pg -triple i386-unknown-unknown -emit-llvm -O2 -o - %s | FileCheck %s
// RUN: %clang_cc1 -pg -triple powerpc-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc64-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc64le-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple i386-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple x86_64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple arm-netbsd-eabi -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple aarch64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mips-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mips-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mipsel-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mips64-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mips64el-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple riscv32-elf -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple riscv64-elf -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple riscv32-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple riscv64-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple riscv64-freebsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple riscv64-freebsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple riscv64-openbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc64le-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple sparc-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple sparc64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s -check-prefix=NO-MCOUNT

int bar(void) {
  return 0;
}

int foo(void) {
  return bar();
}

int __attribute__((no_instrument_function)) no_instrument(void) {
  return foo();
}

int main(void) {
  return no_instrument();
}

// CHECK: call void @mcount
// CHECK: call void @mcount
// CHECK: call void @mcount
// CHECK-NOT: call void @mcount
// CHECK-PREFIXED: call void @_mcount
// CHECK-PREFIXED: call void @_mcount
// CHECK-PREFIXED: call void @_mcount
// CHECK-PREFIXED-NOT: call void @_mcount
// CHECK-DOUBLE-PREFIXED: call void @__mcount
// CHECK-DOUBLE-PREFIXED: call void @__mcount
// CHECK-DOUBLE-PREFIXED: call void @__mcount
// CHECK-DOUBLE-PREFIXED-NOT: call void @__mcount
// NO-MCOUNT-NOT: call void @{{.*}}mcount
// NO-MCOUNT1-NOT: call void @{{.*}}mcount
