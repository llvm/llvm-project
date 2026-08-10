// RUN: %clang_cc1 %s -triple nvptx64 -aux-triple x86_64-unknown-linux-gnu -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fopenmp -fopenmp-is-target-device -triple nvptx64 -aux-triple x86_64-unknown-linux-gnu -E -dM | FileCheck --check-prefix=CHECK-OMP-DEVICE %s

// CHECK-NOT:#define __x86_64__ 1
// CHECK-OMP-DEVICE:#define __x86_64__ 1
