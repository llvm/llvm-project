// RUN: %clang_cc1 -fcuda-is-device \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefixes=NOFTZ,PTXNOFTZ %s

// RUN: %clang_cc1 -fcuda-is-device -fdenormal-fp-math-f32=ieee \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefixes=NOFTZ,PTXNOFTZ %s

// RUN: %clang_cc1 -fcuda-is-device -fdenormal-fp-math-f32=preserve-sign \
// RUN:   -triple nvptx-nvidia-cuda -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefixes=FTZ,PTXFTZ %s

// RUN: %clang_cc1 -fcuda-is-device -x hip \
// RUN:   -triple amdgcn-amd-amdhsa -target-cpu gfx900 -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=NOFTZ %s

// RUN: %clang_cc1 -fcuda-is-device -x hip \
// RUN:   -triple amdgcn-amd-amdhsa -target-cpu gfx900 -fdenormal-fp-math-f32=ieee -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=NOFTZ %s

// RUN: %clang_cc1 -fcuda-is-device -x hip -fdenormal-fp-math-f32=preserve-sign \
// RUN:   -triple amdgcn-amd-amdhsa -target-cpu gfx900 -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=FTZ %s

#include "Inputs/cuda.h"

// Checks that device function calls get emitted with the "denormal-fp-math-f32"
// attribute set when we compile CUDA device code with
// -fdenormal-fp-math-f32. Further, check that we reflect the presence or
// absence of -fgpu-flush-denormals-to-zero in a module flag.

// AMDGCN targets always have f64/f16 denormals enabled.
//
// AMDGCN targets without fast FMAF (e.g. gfx803) always have f32 denormal
// flushing by default.
//
// For AMDGCN target with fast FMAF (e.g. gfx900), it has ieee denormals by
// default and preserve-sign when there with the option
// -fgpu-flush-denormals-to-zero.

// CHECK-LABEL: define void @foo() #0
extern "C" __device__ void foo() {}

// FTZ: attributes #0 = {{.*}} "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// NOFTZ-NOT: "denormal-fp-math-f32"

// PTXFTZ:!llvm.module.flags = !{{{.*}}[[MODFLAG:![0-9]+]]}
// PTXFTZ:[[MODFLAG]] = !{i32 4, !"nvvm-reflect-ftz", i32 1}

// PTXNOFTZ:!llvm.module.flags = !{{{.*}}[[MODFLAG:![0-9]+]]}
// PTXNOFTZ:[[MODFLAG]] = !{i32 4, !"nvvm-reflect-ftz", i32 0}
