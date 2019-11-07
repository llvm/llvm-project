// RUN: %clang_cc1 -fcuda-is-device \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=DEFAULT %s

// RUN: %clang_cc1 -fcuda-is-device -fdenormal-fp-math-f32=ieee \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=NOFTZ %s

// RUN: %clang_cc1 -fcuda-is-device -fdenormal-fp-math-f32=preserve-sign \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=FTZ %s

// FIXME: Unspecified should default to ieee
// RUN: %clang_cc1 -fcuda-is-device -x hip \
// RUN:   -triple amdgcn-amd-amdhsa -target-cpu gfx900 -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=AMDFTZ %s

// RUN: %clang_cc1 -fcuda-is-device -x hip \
// RUN:   -triple amdgcn-amd-amdhsa -target-cpu gfx900 -fdenormal-fp-math-f32=ieee -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=AMDNOFTZ %s

// RUN: %clang_cc1 -fcuda-is-device -x hip -fdenormal-fp-math-f32=preserve-sign \
// RUN:   -triple amdgcn-amd-amdhsa -target-cpu gfx900 -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=AMDFTZ %s

#include "Inputs/cuda.h"

// Checks that device function calls get emitted with the "denormal-fp-math-f32"
// attribute set when we compile CUDA device code with
// -fdenormal-fp-math-f32. Further, check that we reflect the presence or
// absence of -fcuda-flush-denormals-to-zero in a module flag.

// AMDGCN targets always have +fp64-fp16-denormals.
// AMDGCN targets without fast FMAF (e.g. gfx803) always have +fp32-denormals.
// For AMDGCN target with fast FMAF (e.g. gfx900), it has +fp32-denormals
// by default and -fp32-denormals when there is option
// -fcuda-flush-denormals-to-zero.

// CHECK-LABEL: define void @foo() #0
extern "C" __device__ void foo() {}

// FTZ: attributes #0 = {{.*}} "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// NOFTZ: attributes #0 = {{.*}} "denormal-fp-math-f32"="ieee,ieee"


// FIXME: This should be removed
// DEFAULT-NOT: "denormal-fp-math-f32"

// AMDNOFTZ: attributes #0 = {{.*}}+fp32-denormals{{.*}}+fp64-fp16-denormals
// AMDFTZ: attributes #0 = {{.*}}+fp64-fp16-denormals{{.*}}-fp32-denormals

// FTZ:!llvm.module.flags = !{{{.*}}[[MODFLAG:![0-9]+]]}
// FTZ:[[MODFLAG]] = !{i32 4, !"nvvm-reflect-ftz", i32 1}

// NOFTZ:!llvm.module.flags = !{{{.*}}[[MODFLAG:![0-9]+]]}
// NOFTZ:[[MODFLAG]] = !{i32 4, !"nvvm-reflect-ftz", i32 0}
