// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// By default we should fuse multiply/add into fma instruction.
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -disable-llvm-passes -o - %s | FileCheck -check-prefix ENABLED %s

// Explicit -ffp-contract=fast
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -ffp-contract=fast -disable-llvm-passes -o - %s \
// RUN:   | FileCheck -check-prefix ENABLED %s

// Explicit -ffp-contract=on -- fusing by front-end (disabled).
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -ffp-contract=on -disable-llvm-passes -o - %s \
// RUN:   | FileCheck -check-prefix DISABLED %s

// Explicit -ffp-contract=off should disable instruction fusing.
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -S \
// RUN:   -ffp-contract=off -disable-llvm-passes -o - %s \
// RUN:   | FileCheck -check-prefix DISABLED %s


#include "Inputs/cuda.h"

__host__ __device__ float func(float a, float b, float c) { return a + b * c; }
// ENABLED:       fma.rn.f32
// ENABLED-NEXT:  st.param.f32

// DISABLED:      mul.rn.f32
// DISABLED-NEXT: add.rn.f32
// DISABLED-NEXT: st.param.f32
