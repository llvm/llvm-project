// RUN: %clang_cc1 -pg -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -pg -triple i386-unknown-unknown -emit-llvm -O2 -o - %s | FileCheck %s
// RUN: %clang_cc1 -pg -triple powerpc-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple powerpc64-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple powerpc64le-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple i386-netbsd -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple x86_64-netbsd -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple arm-netbsd-eabi -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple aarch64-netbsd -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple mips-netbsd -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple powerpc-netbsd -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple powerpc64-netbsd -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple powerpc64le-netbsd -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple sparc-netbsd -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -pg -triple sparc64-netbsd -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s -check-prefix=NO-MCOUNT

int bar(void) {
  return 0;
}

int foo(void) {
  return bar();
}

int main(void) {
  return foo();
}

// CHECK: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="mcount"{{.*}} }
// CHECK-PREFIXED: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="_mcount"{{.*}} }
// NO-MCOUNT-NOT: attributes #{{[0-9]}} = { {{.*}}"counting-function"={{.*}} }
