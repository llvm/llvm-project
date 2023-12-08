// REQUIRES: arm-registered-target,aarch64-registered-target

// RUN: %clang -target armv7-unknown-none-eabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,UNSUPPORTED
// RUN: %clang -target armv7-unknown-none-eabi -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNSUPPORTED
// RUN: %clang -target aarch64-unknown-none-gnu -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,MCOUNT
// RUN: %clang -target aarch64-unknown-none-gnu -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER
// RUN: %clang -target armv7-unknown-linux-gnueabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI
// RUN: %clang -target armv7-unknown-linux-gnueabi -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-MEABI-GNU
// RUN: %clang -target aarch64-unknown-linux-gnueabi -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER
// RUN: %clang -target aarch64-unknown-linux-gnueabi -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER
// RUN: %clang -target armv7-unknown-linux-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI
// RUN: %clang -target armv7-unknown-linux-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-MEABI-GNU
// RUN: %clang -target aarch64-unknown-linux-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER
// RUN: %clang -target aarch64-unknown-linux-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER
// RUN: %clang -target armv7-unknown-freebsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER_UNDER
// RUN: %clang -target armv7-unknown-freebsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER_UNDER
// RUN: %clang -target aarch64-unknown-freebsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-FREEBSD
// RUN: %clang -target aarch64-unknown-freebsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-FREEBSD
// RUN: %clang -target armv7-unknown-openbsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix UNDER_UNDER
// RUN: %clang -target armv7-unknown-openbsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix UNDER_UNDER
// RUN: %clang -target aarch64-unknown-openbsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix UNDER_UNDER
// RUN: %clang -target aarch64-unknown-openbsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER_UNDER
// RUN: %clang -target armv7-unknown-netbsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER_UNDER
// RUN: %clang -target armv7-unknown-netbsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER_UNDER
// RUN: %clang -target aarch64-unknown-netbsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER_UNDER
// RUN: %clang -target aarch64-unknown-netbsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNDER_UNDER
// RUN: %clang -target armv7-apple-ios -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNSUPPORTED
// RUN: %clang -target armv7-apple-ios -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNSUPPORTED
// RUN: %clang -target arm64-apple-ios -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNSUPPORTED
// RUN: %clang -target arm64-apple-ios -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UNSUPPORTED
// RUN: %clang -target armv7-unknown-rtems-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,MCOUNT
// RUN: %clang -target armv7-unknown-rtems-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,MCOUNT
// RUN: %clang -target aarch64-unknown-rtems-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,MCOUNT
// RUN: %clang -target aarch64-unknown-rtems-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,MCOUNT

int f() {
  return 0;
}

// CHECK-LABEL: f
// TODO: add profiling support for arm-baremetal
// UNSUPPORTED-NOT: call void
// CHECK-ARM-EABI: call void @"\01mcount"()
// MCOUNT: call void @mcount()
// UNDER: call void @"\01_mcount"()
// UNDER_UNDER: call void @__mcount()
// CHECK-ARM64-EABI-FREEBSD: call void @.mcount()
// CHECK-ARM-EABI-MEABI-GNU: call void @llvm.arm.gnu.eabi.mcount()
