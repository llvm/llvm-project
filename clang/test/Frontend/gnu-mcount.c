// REQUIRES: arm-registered-target,aarch64-registered-target

// RUN: %clang -target armv7-unknown-none-eabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-BAREMETAL-EABI
// RUN: %clang -target armv7-unknown-none-eabi -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-BAREMETAL-EABI-MEABI-GNU
// RUN: %clang -target aarch64-unknown-none-eabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-BAREMETAL-EABI
// RUN: %clang -target aarch64-unknown-none-eabi -pg -meabi gnu -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-BAREMETAL-EABI-MEABI-GNU
// RUN: %clang -target armv7-unknown-linux-gnueabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI
// RUN: %clang -target armv7-unknown-linux-gnueabi -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-MEABI-GNU
// RUN: %clang -target aarch64-unknown-linux-gnueabi -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-LINUX
// RUN: %clang -target aarch64-unknown-linux-gnueabi -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-MEABI-GNU
// RUN: %clang -target armv7-unknown-linux-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI
// RUN: %clang -target armv7-unknown-linux-gnueabihf -meabi gnu -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM-EABI-MEABI-GNU
// RUN: %clang -target aarch64-unknown-linux-gnueabihf -pg -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ARM64-EABI-LINUX
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
// TODO: add profiling support for arm-baremetal
// CHECK-ARM-BAREMETAL-EABI-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01mcount"{{.*}} }
// CHECK-ARM-BAREMETAL-EABI-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM64-BAREMETAL-EABI: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="mcount"{{.*}} }
// CHECK-ARM64-BAREMETAL-EABI-MEABI-GNU: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01_mcount"{{.*}} }
// CHECK-ARM-IOS-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="_mcount"{{.*}} }
// CHECK-ARM-IOS-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM-EABI: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01mcount"{{.*}} }
// CHECK-ARM-EABI-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM64-EABI: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="mcount"{{.*}} }
// CHECK-ARM64-EABI-MEABI-GNU: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01_mcount"{{.*}} }
// CHECK-ARM64-EABI-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM64-EABI-LINUX: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01_mcount"{{.*}} }
// CHECK-ARM-EABI-FREEBSD: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="__mcount"{{.*}} }
// CHECK-ARM-EABI-FREEBSD-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM64-EABI-FREEBSD: attributes #{{[0-9]+}} = { {{.*}}"counting-function"=".mcount"{{.*}} }
// CHECK-ARM64-EABI-FREEBSD-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM-EABI-NETBSD: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="_mcount"{{.*}} }
// CHECK-ARM-EABI-NETBSD-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM-EABI-OPENBSD: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="__mcount"{{.*}} }
// CHECK-ARM-EABI-OPENBSD-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM64-EABI-OPENBSD: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="__mcount"{{.*}} }
// CHECK-ARM64-EABI-OPENBSD-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM-EABI-MEABI-GNU-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="mcount"{{.*}} }
// CHECK-ARM-EABI-MEABI-GNU: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM-EABI-BITRIG: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="__mcount"{{.*}} }
// CHECK-ARM-EABI-BITRIG-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM54-EABI-BITRIG: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="mcount"{{.*}} }
// CHECK-ARM54-EABI-BITRIG-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM-EABI-RTEMS: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="mcount"{{.*}} }
// CHECK-ARM-EABI-RTEMS-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM64-EABI-RTEMS: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="mcount"{{.*}} }
// CHECK-ARM64-EABI-RTEMS-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM-EABI-CLOUDABI: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="mcount"{{.*}} }
// CHECK-ARM-EABI-CLOUDABI-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }
// CHECK-ARM64-EABI-CLOUDABI: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="mcount"{{.*}} }
// CHECK-ARM64-EABI-CLOUDABI-NOT: attributes #{{[0-9]+}} = { {{.*}}"counting-function"="\01__gnu_mcount_nc"{{.*}} }

