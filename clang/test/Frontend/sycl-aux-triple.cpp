// RUN: %clang_cc1 %s -triple spir -aux-triple x86_64-unknown-linux-gnu -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl -fsycl-is-device -triple spir -aux-triple x86_64-unknown-linux-gnu -E -dM | FileCheck --check-prefix=CHECK-SYCL %s

// CHECK-NOT:#define __x86_64__ 1
// CHECK-SYCL:#define __x86_64__ 1
