#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target

// Default behavior for gfx90a: test_default has no target-features,
// test_explicit_attr has only the delta (+gfx11-insts).

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -target-cpu gfx90a -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-GFX90A %s --input-file=%t.cir

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -target-cpu gfx90a -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --check-prefix=LLVM-GFX90A %s --input-file=%t.cir.ll

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -target-cpu gfx90a -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-GFX90A %s --input-file=%t.ll

// With -target-feature, both functions get the delta feature.
// gfx1030 defaults to wavefrontsize32, so +wavefrontsize64 is a delta.

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -target-cpu gfx1030 -target-feature +wavefrontsize64 \
// RUN:            -emit-cir %s -o %t-cmdline.cir
// RUN: FileCheck --check-prefix=CIR-CMDLINE %s --input-file=%t-cmdline.cir

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -target-cpu gfx1030 -target-feature +wavefrontsize64 \
// RUN:            -emit-llvm %s -o %t-cmdline.cir.ll
// RUN: FileCheck --check-prefix=LLVM-CMDLINE %s --input-file=%t-cmdline.cir.ll

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -target-cpu gfx1030 -target-feature +wavefrontsize64 \
// RUN:            -emit-llvm %s -o %t-cmdline.ll
// RUN: FileCheck --check-prefix=LLVM-CMDLINE %s --input-file=%t-cmdline.ll

// gfx90a + no extra features: test_default has only target-cpu, no
// target-features. test_explicit_attr has cpu + only the per-function delta.

// CIR-GFX90A:      cir.func{{.*}} @_Z12test_defaultv()
// CIR-GFX90A-SAME: "cir.target-cpu" = "gfx90a"
// CIR-GFX90A-NOT:  "cir.target-features"

// CIR-GFX90A:      cir.func{{.*}} @_Z18test_explicit_attrv()
// CIR-GFX90A-SAME: "cir.target-cpu" = "gfx90a"
// CIR-GFX90A-SAME: "cir.target-features" = "+gfx11-insts"

// LLVM-GFX90A: define{{.*}} @_Z12test_defaultv(){{.*}} #[[ATTR_DEFAULT_90A:[0-9]+]]
// LLVM-GFX90A: define{{.*}} @_Z18test_explicit_attrv(){{.*}} #[[ATTR_EXPLICIT_90A:[0-9]+]]
// LLVM-GFX90A-DAG: attributes #[[ATTR_DEFAULT_90A]] = {{.*}}"target-cpu"="gfx90a"
// LLVM-GFX90A-NOT: attributes #[[ATTR_DEFAULT_90A]] = {{.*}}"target-features"
// LLVM-GFX90A-DAG: attributes #[[ATTR_EXPLICIT_90A]] = {{.*}}"target-cpu"="gfx90a"{{.*}}"target-features"="+gfx11-insts"

// gfx1030 + cmdline +wavefrontsize64: test_default gets the cmdline delta;
// test_explicit_attr gets both the per-function and cmdline deltas.

// CIR-CMDLINE:      cir.func{{.*}} @_Z12test_defaultv()
// CIR-CMDLINE-SAME: "cir.target-cpu" = "gfx1030"
// CIR-CMDLINE-SAME: "cir.target-features" = "+wavefrontsize64"

// CIR-CMDLINE:      cir.func{{.*}} @_Z18test_explicit_attrv()
// CIR-CMDLINE-SAME: "cir.target-cpu" = "gfx1030"
// CIR-CMDLINE-SAME: "cir.target-features" = "{{[^"]*}}+gfx11-insts{{[^"]*}}+wavefrontsize64{{[^"]*}}"

// LLVM-CMDLINE: define{{.*}} @_Z12test_defaultv(){{.*}} #[[ATTR_DEFAULT_1030:[0-9]+]]
// LLVM-CMDLINE: define{{.*}} @_Z18test_explicit_attrv(){{.*}} #[[ATTR_EXPLICIT_1030:[0-9]+]]
// LLVM-CMDLINE-DAG: attributes #[[ATTR_DEFAULT_1030]] = {{.*}}"target-cpu"="gfx1030"{{.*}}"target-features"="+wavefrontsize64"
// LLVM-CMDLINE-DAG: attributes #[[ATTR_EXPLICIT_1030]] = {{.*}}"target-cpu"="gfx1030"{{.*}}"target-features"="{{[^"]*}}+gfx11-insts{{[^"]*}}+wavefrontsize64{{[^"]*}}"

__global__ void test_default() {}

__attribute__((target("gfx11-insts")))
__device__ void test_explicit_attr() {}
