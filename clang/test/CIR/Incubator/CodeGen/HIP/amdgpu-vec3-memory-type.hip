#include "../Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu verde -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu verde -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu verde -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

typedef int v3i32 __attribute__((ext_vector_type(3)));
typedef unsigned int v3u32 __attribute__((ext_vector_type(3)));
typedef float v3f32 __attribute__((ext_vector_type(3)));

//===----------------------------------------------------------------------===//
// Test 96-bit vector
//===----------------------------------------------------------------------===//

// CIR-LABEL: @_Z18test_store_vec3i32
// CIR: cir.store {{.*}} : !cir.vector<!s32i x 3>, !cir.ptr<!cir.vector<!s32i x 3>
// CIR-NOT: x 4
// LLVM-LABEL: define{{.*}} void @_Z18test_store_vec3i32
// LLVM: store <3 x i32>
// LLVM-NOT: <4 x i32>
// OGCG-LABEL: define{{.*}} void @_Z18test_store_vec3i32
// OGCG: store <3 x i32>
__device__ void test_store_vec3i32(v3i32* out, v3i32 val) {
  *out = val;
}

// CIR-LABEL: @_Z17test_load_vec3i32
// CIR: cir.load {{.*}} : !cir.ptr<!cir.vector<!s32i x 3>>, !cir.vector<!s32i x 3>
// CIR-NOT: x 4
// LLVM-LABEL: define{{.*}} void @_Z17test_load_vec3i32
// LLVM: load <3 x i32>
// LLVM-NOT: <4 x i32>
// OGCG-LABEL: define{{.*}} void @_Z17test_load_vec3i32
// OGCG: load <3 x i32>
__device__ void test_load_vec3i32(v3i32* out, v3i32* in) {
  *out = *in;
}

// CIR-LABEL: @_Z18test_store_vec3f32
// CIR: cir.store {{.*}} : !cir.vector<!cir.float x 3>, !cir.ptr<!cir.vector<!cir.float x 3>
// CIR-NOT: x 4
// LLVM-LABEL: define{{.*}} void @_Z18test_store_vec3f32
// LLVM: store <3 x float>
// LLVM-NOT: <4 x float>
// OGCG-LABEL: define{{.*}} void @_Z18test_store_vec3f32
// OGCG: store <3 x float>
__device__ void test_store_vec3f32(v3f32* out, v3f32 val) {
  *out = val;
}

// CIR-LABEL: @_Z17test_load_vec3f32
// CIR: cir.load {{.*}} : !cir.ptr<!cir.vector<!cir.float x 3>>, !cir.vector<!cir.float x 3>
// CIR-NOT: x 4
// LLVM-LABEL: define{{.*}} void @_Z17test_load_vec3f32
// LLVM: load <3 x float>
// LLVM-NOT: <4 x float>
// OGCG-LABEL: define{{.*}} void @_Z17test_load_vec3f32
// OGCG: load <3 x float>
__device__ void test_load_vec3f32(v3f32* out, v3f32* in) {
  *out = *in;
}

// CIR-LABEL: @_Z16test_vec3i32_add
// CIR: cir.binop(add,{{.*}}!cir.vector<!s32i x 3>
// CIR: cir.store {{.*}} : !cir.vector<!s32i x 3>
// CIR-NOT: x 4
// LLVM-LABEL: define{{.*}} void @_Z16test_vec3i32_add
// LLVM: add <3 x i32>
// LLVM: store <3 x i32>
// OGCG-LABEL: define{{.*}} void @_Z16test_vec3i32_add
// OGCG: add{{.*}}<3 x i32>
__device__ void test_vec3i32_add(v3i32* out, v3i32 a, v3i32 b) {
  *out = a + b;
}

// CIR-LABEL: @_Z16test_vec3f32_mul
// CIR: cir.binop(mul,{{.*}}!cir.vector<!cir.float x 3>
// CIR: cir.store {{.*}} : !cir.vector<!cir.float x 3>
// CIR-NOT: x 4
// LLVM-LABEL: define{{.*}} void @_Z16test_vec3f32_mul
// LLVM: fmul{{.*}}<3 x float>
// LLVM: store <3 x float>
// OGCG-LABEL: define{{.*}} void @_Z16test_vec3f32_mul
// OGCG: fmul{{.*}}<3 x float>
__device__ void test_vec3f32_mul(v3f32* out, v3f32 a, v3f32 b) {
  *out = a * b;
}
