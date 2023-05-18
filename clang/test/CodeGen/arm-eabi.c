// REQUIRES: arm-registered-target
// RUN: %clang -target arm-none-eabi -S -o - %s | FileCheck -check-prefix=CHECK-EABI %s
// RUN: %clang -target arm-none-eabi -S -meabi gnu -o - %s | FileCheck -check-prefix=CHECK-GNUEABI %s
// RUN: %clang -target arm-none-eabihf -S -o - %s | FileCheck -check-prefix=CHECK-EABI %s
// RUN: %clang -target arm-none-eabihf -S -meabi gnu -o - %s | FileCheck -check-prefix=CHECK-GNUEABI %s
// RUN: %clang -target arm-none-gnueabi -S -o - %s | FileCheck -check-prefix=CHECK-GNUEABI %s
// RUN: %clang -target arm-none-gnueabi -S -meabi 5 -o - %s | FileCheck -check-prefix=CHECK-EABI %s
// RUN: %clang -target arm-none-gnueabihf -S -o - %s | FileCheck -check-prefix=CHECK-GNUEABI %s
// RUN: %clang -target arm-none-gnueabihf -S -meabi 5 -o - %s | FileCheck -check-prefix=CHECK-EABI %s
// RUN: %clang -target arm-none-musleabi -S -o - %s \
// RUN:   | FileCheck -check-prefix=CHECK-GNUEABI %s
// RUN: %clang -target arm-none-musleabi -S -o - %s -meabi 5 \
// RUN:   | FileCheck -check-prefix=CHECK-EABI %s
// RUN: %clang -target arm-none-musleabihf -S -o - %s \
// RUN:   | FileCheck -check-prefix=CHECK-GNUEABI %s
// RUN: %clang -target arm-none-musleabihf -S -o - %s -meabi 5 \
// RUN:   | FileCheck -check-prefix=CHECK-EABI %s

struct my_s {
  unsigned long a[18];
};

// CHECK-LABEL: foo
// CHECK-EABI: bl __aeabi_memcpy4
// CHECK-GNUEABI: bl memcpy
void foo(unsigned long *t) {
  *(struct my_s *)t = *((struct my_s *)(1UL));
}
