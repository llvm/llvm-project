// RUN: %clang -target armv7-unknown-none-eabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI
// RUN: %clang -target armv7-unknown-none-eabi -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-MEABI-GNU
// RUN: %clang -target aarch64-unknown-none-eabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI
// RUN: %clang -target aarch64-unknown-none-eabi -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-MEABI-GNU
// RUN: %clang -target armv7-unknown-linux-gnueabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI
// RUN: %clang -target armv7-unknown-linux-gnueabi -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-MEABI-GNU
// RUN: %clang -target aarch64-unknown-linux-gnueabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI
// RUN: %clang -target aarch64-unknown-linux-gnueabi -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-MEABI-GNU
// RUN: %clang -target armv7-unknown-linux-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI
// RUN: %clang -target armv7-unknown-linux-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-MEABI-GNU
// RUN: %clang -target aarch64-unknown-linux-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI
// RUN: %clang -target aarch64-unknown-linux-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-MEABI-GNU
// RUN: %clang -target armv7-unknown-freebsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-FREEBSD
// RUN: %clang -target armv7-unknown-freebsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-FREEBSD
// RUN: %clang -target aarch64-unknown-freebsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-FREEBSD
// RUN: %clang -target aarch64-unknown-freebsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-FREEBSD
// RUN: %clang -target armv7-unknown-openbsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-OPENBSD
// RUN: %clang -target armv7-unknown-openbsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-OPENBSD
// RUN: %clang -target aarch64-unknown-openbsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-OPENBSD
// RUN: %clang -target aarch64-unknown-openbsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-OPENBSD
// RUN: %clang -target armv7-unknown-netbsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-NETBSD
// RUN: %clang -target armv7-unknown-netbsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-NETBSD
// RUN: %clang -target aarch64-unknown-netbsd-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-NETBSD
// RUN: %clang -target aarch64-unknown-netbsd-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-NETBSD
// RUN: %clang -target armv7-apple-ios -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-IOS
// RUN: %clang -target armv7-apple-ios -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-IOS
// RUN: %clang -target arm64-apple-ios -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-IOS
// RUN: %clang -target arm64-apple-ios -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-IOS
// RUN: %clang -target armv7-unknown-bitrig-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-BIGRIG
// RUN: %clang -target armv7-unknown-bitrig-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-BIGRIG
// RUN: %clang -target aarch64-unknown-bitrig-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-BITRIG
// RUN: %clang -target aarch64-unknown-bitrig-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-BITRIG
// RUN: %clang -target armv7-unknown-rtems-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-RTEMS
// RUN: %clang -target armv7-unknown-rtems-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-RTEMS
// RUN: %clang -target aarch64-unknown-rtems-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-RTEMS
// RUN: %clang -target aarch64-unknown-rtems-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-RTEMS
// RUN: %clang -target armv7-unknown-cloudabi-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-CLOUDABI
// RUN: %clang -target armv7-unknown-cloudabi-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-CLOUDABI
// RUN: %clang -target aarch64-unknown-cloudabi-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-CLOUDABI
// RUN: %clang -target aarch64-unknown-cloudabi-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-CLOUDABI

int f() {
  return 0;
}

// CHECK-LABEL: f
// CHECK-ARM-IOS-NOT: call void @_mcount()
// CHECK-ARM-IOS-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM-EABI: call void @"\01mcount"()
// CHECK-ARM-EABI-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM64-EABI: call void @mcount()
// CHECK-ARM64-EABI-MEABI-GNU: call void @"\01_mcount"()
// CHECK-ARM64-EABI-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM-EABI-FREEBSD: call void @__mcount()
// CHECK-ARM-EABI-FREEBSD-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM64-EABI-FREEBSD: call void @.mcount()
// CHECK-ARM64-EABI-FREEBSD-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM-EABI-NETBSD: call void @_mcount()
// CHECK-ARM-EABI-NETBSD-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM-EABI-OPENBSD: call void @__mcount()
// CHECK-ARM-EABI-OPENBSD-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM64-EABI-OPENBSD: call void @mcount()
// CHECK-ARM64-EABI-OPENBSD-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM-EABI-MEABI-GNU-NOT: call void @mcount()
// CHECK-ARM-EABI-MEABI-GNU: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM-EABI-BITRIG: call void @__mcount()
// CHECK-ARM-EABI-BITRIG-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM54-EABI-BITRIG: call void @mcount()
// CHECK-ARM54-EABI-BITRIG-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM-EABI-RTEMS: call void @mcount()
// CHECK-ARM-EABI-RTEMS-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM64-EABI-RTEMS: call void @mcount()
// CHECK-ARM64-EABI-RTEMS-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM-EABI-CLOUDABI: call void @mcount()
// CHECK-ARM-EABI-CLOUDABI-NOT: call void @"\01__gnu_mcount_nc"()
// CHECK-ARM64-EABI-CLOUDABI: call void @mcount()
// CHECK-ARM64-EABI-CLOUDABI-NOT: call void @"\01__gnu_mcount_nc"()

