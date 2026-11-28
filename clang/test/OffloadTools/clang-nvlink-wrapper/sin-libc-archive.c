// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -gpulibc -O2 -emit-llvm-bc %s -o %t-user.bc
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -DMOCK_SIN -emit-llvm-bc %s -o %t-lib.bc
// RUN: llvm-ar rcs %t-lib.a %t-lib.bc
// RUN: clang-nvlink-wrapper --lto-emit-llvm -arch sm_70 %t-user.bc %t-lib.a -o %t-linked.bc
// RUN: llvm-dis %t-linked.bc -o - | FileCheck %s --check-prefix=LINKED
// RUN: clang-nvlink-wrapper --lto-emit-asm -arch sm_70 %t-user.bc %t-lib.a -o %t.ptx
// RUN: FileCheck %s --check-prefix=PTX < %t.ptx

#ifdef MOCK_SIN
double sin(double x) { return x; }
#else
extern double sin(double);

// LINKED-LABEL: define{{.*}} @ordinary_sin(
// LINKED: call{{.*}} @sin(
double ordinary_sin(double x) { return sin(x); }

// LINKED-LABEL: define{{.*}} @builtin_sin(
// LINKED: call{{.*}} @sin(
double builtin_sin(double x) { return __builtin_sin(x); }

// LINKED-LABEL: define{{.*}} @sin(

// PTX-COUNT-2: call.uni {{.*}}sin
#endif
