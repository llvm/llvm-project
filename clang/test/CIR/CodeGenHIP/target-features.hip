#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -target-cpu gfx900 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -target-cpu gfx900 -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.cir.ll

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -target-cpu gfx900 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ll

// Verify that with extra target-features differing from the CPU baseline,
// only the delta (differing features) is emitted on cir.target-features.

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -target-cpu gfx900 -target-feature +xnack \
// RUN:            -emit-cir %s -o %t-delta.cir
// RUN: FileCheck --check-prefix=CIR-DELTA %s --input-file=%t-delta.cir

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -target-cpu gfx900 -target-feature +xnack \
// RUN:            -emit-llvm %s -o %t-delta.cir.ll
// RUN: FileCheck --check-prefix=LLVM-DELTA %s --input-file=%t-delta.cir.ll

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -target-cpu gfx900 -target-feature +xnack \
// RUN:            -emit-llvm %s -o %t-delta.ll
// RUN: FileCheck --check-prefix=LLVM-DELTA %s --input-file=%t-delta.ll

// AMDGPU with gfx900: target-cpu is set, target-features is absent.

// CIR: cir.func{{.*}} @_Z6kernelv()
// CIR-SAME: "cir.target-cpu" = "gfx900"
// CIR-NOT: cir.target-features
__global__ void kernel() {}

// CIR: cir.func{{.*}} @_Z9device_fnv()
// CIR-SAME: "cir.target-cpu" = "gfx900"
// CIR-NOT: cir.target-features
__device__ void device_fn() {}

// LLVM: define{{.*}} void @_Z6kernelv(){{.*}} #[[K_ATTR:[0-9]+]]
// LLVM: define{{.*}} void @_Z9device_fnv(){{.*}} #[[D_ATTR:[0-9]+]]
// LLVM-DAG: attributes #[[K_ATTR]] = {{.*}}"target-cpu"="gfx900"
// LLVM-DAG: attributes #[[D_ATTR]] = {{.*}}"target-cpu"="gfx900"

// AMDGPU with gfx900 + an extra +xnack feature: only the delta is emitted.

// CIR-DELTA: cir.func{{.*}} @_Z6kernelv()
// CIR-DELTA-SAME: "cir.target-cpu" = "gfx900"
// CIR-DELTA-SAME: "cir.target-features" = "+xnack"

// CIR-DELTA: cir.func{{.*}} @_Z9device_fnv()
// CIR-DELTA-SAME: "cir.target-cpu" = "gfx900"
// CIR-DELTA-SAME: "cir.target-features" = "+xnack"

// LLVM-DELTA: define{{.*}} void @_Z6kernelv(){{.*}} #[[K_ATTR_D:[0-9]+]]
// LLVM-DELTA: define{{.*}} void @_Z9device_fnv(){{.*}} #[[D_ATTR_D:[0-9]+]]
// LLVM-DELTA-DAG: attributes #[[K_ATTR_D]] = {{.*}}"target-cpu"="gfx900"{{.*}}"target-features"="+xnack"
// LLVM-DELTA-DAG: attributes #[[D_ATTR_D]] = {{.*}}"target-cpu"="gfx900"{{.*}}"target-features"="+xnack"
