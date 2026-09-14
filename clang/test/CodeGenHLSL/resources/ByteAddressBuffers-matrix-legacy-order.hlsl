// RUN: %clang_cc1 -triple spirv-unknown-vulkan-library -finclude-default-header \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-library -finclude-default-header \
// RUN:   -fspv-use-legacy-buffer-matrix-order -emit-llvm -disable-llvm-passes -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,LEGACY

// Raw buffers carry no layout information. By default, a matrix read from or
// written to a raw buffer is assumed to be stored in column-major order,
// matching the native in-register representation, so no reordering is
// needed. -fspv-use-legacy-buffer-matrix-order assumes the raw bytes are
// stored in row-major order instead, which requires transposing the loaded
// (or, before storing, the to-be-stored) value.

ByteAddressBuffer Buf : register(t0);
RWByteAddressBuffer RWBuf : register(u0);

export float2x3 TestLoad() {
  return Buf.Load<float2x3>(0);
}

// CHECK-LABEL: define {{.*}} <6 x float> @{{.*}}ByteAddressBuffer4LoadIu11matrix_typeILj2ELj3EfEEET_j
// CHECK: [[LOADED:%.*]] = load <6 x float>, ptr addrspace(11) %{{.*}}
// DEFAULT-NOT: call {{.*}} @llvm.matrix.transpose
// DEFAULT: ret <6 x float> [[LOADED]]
// LEGACY: [[TRANSPOSED:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[LOADED]], i32 3, i32 2)
// LEGACY: ret <6 x float> [[TRANSPOSED]]

export void TestStore(float2x3 M) {
  RWBuf.Store<float2x3>(0, M);
}

// CHECK-LABEL: define {{.*}} void @{{.*}}RWByteAddressBuffer5StoreIu11matrix_typeILj2ELj3EfEEEvjT_
// CHECK: [[VALUE:%.*]] = load <6 x float>, ptr %Value.addr
// DEFAULT-NOT: call {{.*}} @llvm.matrix.transpose
// DEFAULT: store <6 x float> [[VALUE]], ptr addrspace(11) %{{.*}}
// LEGACY: [[TRANSPOSED:%.*]] = call {{.*}} <6 x float> @llvm.matrix.transpose.v6f32(<6 x float> [[VALUE]], i32 2, i32 3)
// LEGACY: store <6 x float> [[TRANSPOSED]], ptr addrspace(11) %{{.*}}
