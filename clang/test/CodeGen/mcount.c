// RUN: %clang_cc1 -disable-llvm-passes -pg -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple i386-unknown-unknown -emit-llvm -O2 -o - %s | FileCheck %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple powerpc-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple powerpc64-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple powerpc64le-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple i386-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple x86_64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple arm-netbsd-eabi -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple aarch64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple loongarch32 -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple loongarch64 -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple mips-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple mips-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple mipsel-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple mips64-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple mips64el-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple riscv32-elf -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple riscv64-elf -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple riscv32-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple riscv64-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple riscv64-freebsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple riscv64-freebsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple riscv64-openbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple powerpc-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple powerpc64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple powerpc64le-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple sparc-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple sparc64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s -check-prefix=NO-MCOUNT

int bar(void) {
// CHECK: define dso_local i32 @bar() #0 {
  return 0;
}

int foo(void) {
// CHECK: define dso_local i32 @foo() #0 {
  return bar();
}

int __attribute__((no_instrument_function)) no_instrument(void) {
// CHECK: define dso_local i32 @no_instrument() #1 {
  return foo();
}

int main(void) {
// CHECK: define dso_local i32 @main() #0 {
  return no_instrument();
}

// CHECK: attributes #0 = { {{.*}} "instrument-function-entry-inlined"="mcount"
// CHECK-NOT: attributes #1 = { {{.*}}"mcount"
// CHECK-PREFIXED: attributes #0 = { {{.*}} "instrument-function-entry-inlined"="_mcount"
// CHECK-PREFIXED-NOT: attributes #1 = { {{.*}}"_mcount"
// CHECK-DOUBLE-PREFIXED: attributes #0 = { {{.*}} "instrument-function-entry-inlined"="__mcount"
// CHECK-DOUBLE-PREFIXED-NOT: attributes #1 = { {{.*}}"__mcount"
// NO-MCOUNT-NOT: attributes{{.*}}mcount
// NO-MCOUNT1-NOT: attributes{{.*}}mcount
