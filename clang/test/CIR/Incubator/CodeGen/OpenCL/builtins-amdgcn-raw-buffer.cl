// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu verde -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:            -target-cpu verde -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -cl-std=CL2.0 \
// RUN:            -target-cpu verde -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

//===----------------------------------------------------------------------===//
// Test raw buffer load/store builtins
//===----------------------------------------------------------------------===//

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned int v2u32 __attribute__((ext_vector_type(2)));
typedef unsigned int v3u32 __attribute__((ext_vector_type(3)));
typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));

// CIR-LABEL: @test_raw_buffer_store_b8
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.store" {{.*}} : (!u8i, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i)
// LLVM-LABEL: define{{.*}} void @test_raw_buffer_store_b8
// LLVM: call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} void @test_raw_buffer_store_b8
// OGCG: call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
void test_raw_buffer_store_b8(u8 vdata, __amdgpu_buffer_rsrc_t rsrc) {
  __builtin_amdgcn_raw_buffer_store_b8(vdata, rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_store_b16
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.store" {{.*}} : (!u16i, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i)
// LLVM-LABEL: define{{.*}} void @test_raw_buffer_store_b16
// LLVM: call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} void @test_raw_buffer_store_b16
// OGCG: call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
void test_raw_buffer_store_b16(u16 vdata, __amdgpu_buffer_rsrc_t rsrc) {
  __builtin_amdgcn_raw_buffer_store_b16(vdata, rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_store_b32
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.store" {{.*}} : (!u32i, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i)
// LLVM-LABEL: define{{.*}} void @test_raw_buffer_store_b32
// LLVM: call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} void @test_raw_buffer_store_b32
// OGCG: call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
void test_raw_buffer_store_b32(u32 vdata, __amdgpu_buffer_rsrc_t rsrc) {
  __builtin_amdgcn_raw_buffer_store_b32(vdata, rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_store_b64
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.store" {{.*}} : (!cir.vector<!u32i x 2>, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i)
// LLVM-LABEL: define{{.*}} void @test_raw_buffer_store_b64
// LLVM: call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} void @test_raw_buffer_store_b64
// OGCG: call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
void test_raw_buffer_store_b64(v2u32 vdata, __amdgpu_buffer_rsrc_t rsrc) {
  __builtin_amdgcn_raw_buffer_store_b64(vdata, rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_store_b96
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.store" {{.*}} : (!cir.vector<!u32i x 3>, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i)
// LLVM-LABEL: define{{.*}} void @test_raw_buffer_store_b96
// LLVM: call void @llvm.amdgcn.raw.ptr.buffer.store.v3i32(<3 x i32> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} void @test_raw_buffer_store_b96
// OGCG: call void @llvm.amdgcn.raw.ptr.buffer.store.v3i32(<3 x i32> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
void test_raw_buffer_store_b96(v3u32 vdata, __amdgpu_buffer_rsrc_t rsrc) {
  __builtin_amdgcn_raw_buffer_store_b96(vdata, rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_store_b128
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.store" {{.*}} : (!cir.vector<!u32i x 4>, !cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i)
// LLVM-LABEL: define{{.*}} void @test_raw_buffer_store_b128
// LLVM: call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} void @test_raw_buffer_store_b128
// OGCG: call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %{{.*}}, ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
void test_raw_buffer_store_b128(v4u32 vdata, __amdgpu_buffer_rsrc_t rsrc) {
  __builtin_amdgcn_raw_buffer_store_b128(vdata, rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_load_b8
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.load" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !u8i
// LLVM-LABEL: define{{.*}} i8 @test_raw_buffer_load_b8
// LLVM: call i8 @llvm.amdgcn.raw.ptr.buffer.load.i8(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} i8 @test_raw_buffer_load_b8
// OGCG: call i8 @llvm.amdgcn.raw.ptr.buffer.load.i8(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
u8 test_raw_buffer_load_b8(__amdgpu_buffer_rsrc_t rsrc) {
  return __builtin_amdgcn_raw_buffer_load_b8(rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_load_b16
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.load" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !u16i
// LLVM-LABEL: define{{.*}} i16 @test_raw_buffer_load_b16
// LLVM: call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} i16 @test_raw_buffer_load_b16
// OGCG: call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
u16 test_raw_buffer_load_b16(__amdgpu_buffer_rsrc_t rsrc) {
  return __builtin_amdgcn_raw_buffer_load_b16(rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_load_b32
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.load" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !u32i
// LLVM-LABEL: define{{.*}} i32 @test_raw_buffer_load_b32
// LLVM: call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} i32 @test_raw_buffer_load_b32
// OGCG: call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
u32 test_raw_buffer_load_b32(__amdgpu_buffer_rsrc_t rsrc) {
  return __builtin_amdgcn_raw_buffer_load_b32(rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_load_b64
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.load" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.vector<!u32i x 2>
// LLVM-LABEL: define{{.*}} <2 x i32> @test_raw_buffer_load_b64
// LLVM: call <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} <2 x i32> @test_raw_buffer_load_b64
// OGCG: call <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
v2u32 test_raw_buffer_load_b64(__amdgpu_buffer_rsrc_t rsrc) {
  return __builtin_amdgcn_raw_buffer_load_b64(rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_load_b96
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.load" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.vector<!u32i x 3>
// LLVM-LABEL: define{{.*}} <3 x i32> @test_raw_buffer_load_b96
// LLVM: call <3 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v3i32(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} <3 x i32> @test_raw_buffer_load_b96
// OGCG: call <3 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v3i32(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
v3u32 test_raw_buffer_load_b96(__amdgpu_buffer_rsrc_t rsrc) {
  return __builtin_amdgcn_raw_buffer_load_b96(rsrc, 0, 0, 0);
}

// CIR-LABEL: @test_raw_buffer_load_b128
// CIR: cir.llvm.intrinsic "amdgcn.raw.ptr.buffer.load" {{.*}} : (!cir.ptr<!void, target_address_space(8)>, !s32i, !s32i, !s32i) -> !cir.vector<!u32i x 4>
// LLVM-LABEL: define{{.*}} <4 x i32> @test_raw_buffer_load_b128
// LLVM: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
// OGCG-LABEL: define{{.*}} <4 x i32> @test_raw_buffer_load_b128
// OGCG: call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %{{.*}}, i32 0, i32 0, i32 0)
v4u32 test_raw_buffer_load_b128(__amdgpu_buffer_rsrc_t rsrc) {
  return __builtin_amdgcn_raw_buffer_load_b128(rsrc, 0, 0, 0);
}
