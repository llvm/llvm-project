// RUN: %clang_cc1 -fcuda-is-device \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck %s -check-prefix CHECK -check-prefix NOFTZ
// RUN: %clang_cc1 -fcuda-is-device -fcuda-flush-denormals-to-zero \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck %s -check-prefix CHECK -check-prefix FTZ

#include "Inputs/cuda.h"

// Checks that device function calls get emitted with the "ntpvx-f32ftz"
// attribute set to "true" when we compile CUDA device code with
// -fcuda-flush-denormals-to-zero.  Further, check that we reflect the presence
// or absence of -fcuda-flush-denormals-to-zero in a module flag.

// CHECK-LABEL: define void @foo() #0
extern "C" __device__ void foo() {}

// FTZ: attributes #0 = {{.*}} "nvptx-f32ftz"="true"
// NOFTZ-NOT: attributes #0 = {{.*}} "nvptx-f32ftz"

// FTZ:!llvm.module.flags = !{{{.*}}[[MODFLAG:![0-9]+]]}
// FTZ:[[MODFLAG]] = !{i32 4, !"nvvm-reflect-ftz", i32 1}

// NOFTZ:!llvm.module.flags = !{{{.*}}[[MODFLAG:![0-9]+]]}
// NOFTZ:[[MODFLAG]] = !{i32 4, !"nvvm-reflect-ftz", i32 0}
