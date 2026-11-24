// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -finclude-default-header -emit-llvm -O3 -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -O3 -o - %s | FileCheck %s --check-prefixes=CHECK,DXIL

// Check that raw buffer load/stores are lowered to appropriate intrinsics, and aggregate
// accesses are split into accesses of individual members/elements

ByteAddressBuffer input : register(t0);
RWByteAddressBuffer output : register(u1);
StructuredBuffer<uint> structured : register(t2);


// CHECK-LABEL: define {{.*}} float @{{.*}}load_scalar
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, float{{.*}} %{{.*}}, i32 4)
// DXIL: call { float, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call void @llvm.dx.resource.store.rawbuffer
export float load_scalar(uint offset) {
  float value = input.Load<float>(offset);
  output.Store<float>(offset, value);
  return value;
}

// CHECK-LABEL: define {{.*}}void @{{.*}}copy_vector
// SPIRV: %[[V:.*]] = {{.*}}call {{.*}}<4 x i32> @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, <4 x i32> %[[V]], i32 4)
// DXIL: call { <4 x i32>, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call void @llvm.dx.resource.store.rawbuffer
export void copy_vector(uint offset) {
  output.Store4(offset, input.Load4(offset));
}

struct Data {
  float x;
  uint2 y;
};

// load/stores of structs should be split into separate raw buffer intrinsics per vector or
// scalar member

// CHECK-LABEL: define {{.*}}void @{{.*}}copy_struct
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer
// SPIRV: call {{.*}}<2 x i32> @llvm.spv.resource.load.rawbuffer
// SPIRV: call void @llvm.spv.resource.store.rawbuffer
// DXIL: call { float, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call { <2 x i32>, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call void @llvm.dx.resource.store.rawbuffer
export void copy_struct(uint offset) {
  output.Store<Data>(offset, input.Load<Data>(offset));
}

// CHECK-LABEL: define {{.*}}void @{{.*}}copy_bool
// SPIRV: call {{.*}}i32 @llvm.spv.resource.load.rawbuffer
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, i32 4)
// DXIL: call { i32, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call void @llvm.dx.resource.store.rawbuffer
export void copy_bool(uint offset) {
  output.Store<bool>(offset, input.Load<bool>(offset));
}

// CHECK-LABEL: define {{.*}}void @{{.*}}copy_matrix
// SPIRV: call {{.*}}<6 x float> @llvm.spv.resource.load.rawbuffer
// SPIRV: call void @llvm.spv.resource.store.rawbuffer
// DXIL: call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call { <2 x float>, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call void @llvm.dx.resource.store.rawbuffer{{.*}}.v4f32
// DXIL: call void @llvm.dx.resource.store.rawbuffer{{.*}}.v2f32
export void copy_matrix(uint offset) {
  output.Store<float2x3>(offset, input.Load<float2x3>(offset));
}

// Don't use raw buffer intrinsics for structured accesses

// CHECK-LABEL: define {{.*}}i32 @{{.*}}load_structured
// DXIL: call {{.*}}ptr @llvm.dx.resource.getpointer
// SPIRV: call {{.*}}ptr addrspace(11) @llvm.spv.resource.getpointer
// CHECK: load i32
export uint load_structured(uint index) {
  return structured[index];
}

// SPIRV-LABEL: define {{.*}}void @{{.*}}copy_double
// SPIRV: call {{.*}}double @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, i32 8)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, double{{.*}} %{{.*}}, i32 8)
export void copy_double(uint offset) {
  output.Store<double>(offset, input.Load<double>(offset));
}

// SPIRV-LABEL: define {{.*}}void @{{.*}}copy_double_vector
// SPIRV: call {{.*}}<4 x double> @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, i32 8)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, <4 x double>{{.*}} %{{.*}}, i32 8)
export void copy_double_vector(uint offset) {
  output.Store<double4>(offset, input.Load<double4>(offset));
}

struct Mixed {
  float a;
  float b;
  uint64_t c; // should be 8-byte aligned
};

// SPIRV-LABEL: define {{.*}}void @{{.*}}copy_mixed
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, i32 8)
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, i32 4)
// SPIRV: call {{.*}}i64 @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, i32 8)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, float{{.*}} %{{.*}}, i32 8)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, float{{.*}} %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i64 %{{.*}}, i32 8)
export void copy_mixed(uint offset) {
  output.Store<Mixed>(offset, input.Load<Mixed>(offset));
}

struct Arrays {
  float values[2][2];
  Data records[2];
};

// Nested arrays and arrays of structs should be split into separate raw buffer intrinisics
// per element

// CHECK-LABEL: define {{.*}}void @{{.*}}copy_arrays
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 0, i32 4)
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 4, i32 4)
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 8, i32 4)
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 12, i32 4)
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 16, i32 4)
// SPIRV: call {{.*}}<2 x i32> @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 20, i32 4)
// SPIRV: call {{.*}}float @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 28, i32 4)
// SPIRV: call {{.*}}<2 x i32> @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 32, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 0, float{{.*}} %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 4, float{{.*}} %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 8, float{{.*}} %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 12, float{{.*}} %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 16, float{{.*}} %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 20, <2 x i32> %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 28, float{{.*}} %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 32, <2 x i32> %{{.*}}, i32 4)
// DXIL-COUNT-5: call { float, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call { <2 x i32>, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call { float, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call { <2 x i32>, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL-COUNT-8: call void @llvm.dx.resource.store.rawbuffer
export void copy_arrays() {
  output.Store<Arrays>(0, input.Load<Arrays>(0));
}

// CHECK-LABEL: define {{.*}}void @{{.*}}copy_top_level_array
// SPIRV-COUNT-2: call {{.*}}float @llvm.spv.resource.load.rawbuffer
// SPIRV-COUNT-2: call void @llvm.spv.resource.store.rawbuffer
// DXIL-COUNT-2: call { float, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL-COUNT-2: call void @llvm.dx.resource.store.rawbuffer
export void copy_top_level_array(uint offset) {
  output.Store<float[2]>(offset, input.Load<float[2]>(offset));
}

// Load/store a long vector. In DXIL path, clang should do the splitting,
// in spirv, defer that decision and emit 1 9-element load/store

// CHECK-LABEL: define {{.*}}void @{{.*}}copy_long_vector
// SPIRV: %[[LONG:.*]] = {{.*}}call {{.*}}<9 x float> @llvm.spv.resource.load.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, i32 4)
// SPIRV: call void @llvm.spv.resource.store.rawbuffer{{.*}}({{.*}}, i32 %{{.*}}, <9 x float>{{.*}} %[[LONG]], i32 4)
// DXIL-COUNT-2: call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL: call { <1 x float>, i1 } @llvm.dx.resource.load.rawbuffer
// DXIL-COUNT-2: call void @llvm.dx.resource.store.rawbuffer{{.*}}.v4f32
// DXIL: call void @llvm.dx.resource.store.rawbuffer{{.*}}.v1f32
export void copy_long_vector(uint offset) {
  output.Store<vector<float, 9>>(offset, input.Load<vector<float, 9>>(offset));
}
