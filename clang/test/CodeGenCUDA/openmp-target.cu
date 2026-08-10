// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm \
// RUN:            -fopenmp -fopenmp-version=50 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm \
// RUN:            -fopenmp -fopenmp-version=50 -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:            -emit-llvm -o - %s | FileCheck -check-prefixes=DEV %s

// CHECK: declare{{.*}}@_Z7nohost1v()
// DEV-NOT: _Z7nohost1v
void nohost1() {}
#pragma omp declare target to(nohost1) device_type(nohost)

// CHECK: declare{{.*}}@_Z7nohost2v()
// DEV-NOT: _Z7nohost2v
void nohost2() {nohost1();}
#pragma omp declare target to(nohost2) device_type(nohost)

