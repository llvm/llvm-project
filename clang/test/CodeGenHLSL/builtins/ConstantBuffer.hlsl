// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -triple spirv-vulkan-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

struct S {
  float a;
  int b;
};

// CHECK-DXIL: %"class.hlsl::ConstantBuffer" = type { target("dx.CBuffer", %S) }
// CHECK-SPIRV: %"class.hlsl::ConstantBuffer" = type { target("spirv.VulkanBuffer", %S, 2, 0) }
ConstantBuffer<S> cb;

// CHECK-LABEL: define {{.*}} void @_Z4mainv()
// CHECK-DXIL: [[CB_CONV:%.*]] = call noundef {{.*}} ptr addrspace(2) @_ZNK4hlsl14ConstantBufferI1SEcvRU3AS2S1_Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZL2cb)
// CHECK-DXIL: [[GEP_A:%.*]] = getelementptr inbounds nuw %S, ptr addrspace(2) [[CB_CONV]], i32 0, i32 0
// CHECK-DXIL: [[LOAD_A:%.*]] = load float, ptr addrspace(2) [[GEP_A]], align 4

// CHECK-SPIRV: [[CB_CONV:%.*]] = call noundef {{.*}} ptr addrspace(12) @_ZNK4hlsl14ConstantBufferI1SEcvRU4AS12S1_Ev(ptr noundef nonnull align 8 dereferenceable(8) @_ZL2cb)
// CHECK-SPIRV: [[GEP_A:%.*]] = getelementptr inbounds nuw %S, ptr addrspace(12) [[CB_CONV]], i32 0, i32 0
// CHECK-SPIRV: [[LOAD_A:%.*]] = load float, ptr addrspace(12) [[GEP_A]], align 4

// CHECK: store float [[LOAD_A]], ptr %f, align 4
[numthreads(1,1,1)]
void main() {
  float f = cb.a;
}

struct Nested {
  S s;
  float c;
};

ConstantBuffer<Nested> cb_nested[2];

[numthreads(1,1,1)]
void foo() {
  // CHECK-LABEL: define {{.*}} void @_Z3foov()
  // CHECK-DXIL: [[TMP_CB:%.*]] = alloca %"class.hlsl::ConstantBuffer.0", align 4
  // CHECK-DXIL: call void @_ZN4hlsl14ConstantBufferI6NestedE27__createFromImplicitBindingEjjijPKc(ptr dead_on_unwind writable sret(%"class.hlsl::ConstantBuffer.0") align 4 [[TMP_CB]], i32 noundef 1, i32 noundef 0, i32 noundef 2, i32 noundef 1, ptr noundef @cb_nested.str)
  // CHECK-DXIL: [[CB_CONV:%.*]] = call noundef {{.*}} ptr addrspace(2) @_ZNK4hlsl14ConstantBufferI6NestedEcvRU3AS2S1_Ev(ptr noundef nonnull align 4 dereferenceable(4) [[TMP_CB]])
  // CHECK-DXIL: [[GEP_S:%.*]] = getelementptr inbounds nuw %Nested, ptr addrspace(2) [[CB_CONV]], i32 0, i32 0
  // CHECK-DXIL: [[GEP_A2:%.*]] = getelementptr inbounds nuw %S, ptr addrspace(2) [[GEP_S]], i32 0, i32 0
  // CHECK-DXIL: [[LOAD_A2:%.*]] = load float, ptr addrspace(2) [[GEP_A2]], align 4

  // CHECK-SPIRV: [[TMP_CB:%.*]] = alloca %"class.hlsl::ConstantBuffer.0", align 8
  // CHECK-SPIRV: call void @_ZN4hlsl14ConstantBufferI6NestedE27__createFromImplicitBindingEjjijPKc(ptr dead_on_unwind writable sret(%"class.hlsl::ConstantBuffer.0") align 8 [[TMP_CB]], i32 noundef 1, i32 noundef 0, i32 noundef 2, i32 noundef 1, ptr noundef @cb_nested.str)
  // CHECK-SPIRV: [[CB_CONV:%.*]] = call noundef {{.*}} ptr addrspace(12) @_ZNK4hlsl14ConstantBufferI6NestedEcvRU4AS12S1_Ev(ptr noundef nonnull align 8 dereferenceable(8) [[TMP_CB]])
  // CHECK-SPIRV: [[GEP_S:%.*]] = getelementptr inbounds nuw %Nested, ptr addrspace(12) [[CB_CONV]], i32 0, i32 0
  // CHECK-SPIRV: [[GEP_A2:%.*]] = getelementptr inbounds nuw %S, ptr addrspace(12) [[GEP_S]], i32 0, i32 0
  // CHECK-SPIRV: [[LOAD_A2:%.*]] = load float, ptr addrspace(12) [[GEP_A2]], align 4

  // CHECK: store float [[LOAD_A2]], ptr %f2, align 4
  float f2 = cb_nested[1].s.a;
}

void takes_s(S s) {}
void takes_cb(ConstantBuffer<S> c) {}

[numthreads(1,1,1)]
void test_assignments_and_params() {
  // CHECK-LABEL: define {{.*}} void @_Z27test_assignments_and_paramsv()
  
  // CHECK-DXIL: [[CB_CONV1:%.*]] = call noundef {{.*}} ptr addrspace(2) @_ZNK4hlsl14ConstantBufferI1SEcvRU3AS2S1_Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZL2cb)
  // CHECK-DXIL: [[CB_AS1:%.*]] = addrspacecast ptr addrspace(2) [[CB_CONV1]] to ptr
  // CHECK-DXIL: call void @llvm.memcpy.p0.p0.i32(ptr align 1 %s, ptr align 1 [[CB_AS1]], i32 8, i1 false)
  // CHECK-SPIRV: [[CB_CONV1:%.*]] = call noundef {{.*}} ptr addrspace(12) @_ZNK4hlsl14ConstantBufferI1SEcvRU4AS12S1_Ev(ptr noundef nonnull align 8 dereferenceable(8) @_ZL2cb)
  // CHECK-SPIRV: [[CB_AS1:%.*]] = addrspacecast ptr addrspace(12) [[CB_CONV1]] to ptr
  // CHECK-SPIRV: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %s, ptr align 1 [[CB_AS1]], i64 8, i1 false)
  S s = cb;

  // CHECK-DXIL: [[CB_CONV2:%.*]] = call noundef {{.*}} ptr addrspace(2) @_ZNK4hlsl14ConstantBufferI1SEcvRU3AS2S1_Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZL2cb)
  // CHECK-DXIL: [[CB_AS2:%.*]] = addrspacecast ptr addrspace(2) [[CB_CONV2]] to ptr
  // CHECK-DXIL: call void @llvm.memcpy.p0.p0.i32(ptr align 1 %s, ptr align 1 [[CB_AS2]], i32 8, i1 false)
  // CHECK-SPIRV: [[CB_CONV2:%.*]] = call noundef {{.*}} ptr addrspace(12) @_ZNK4hlsl14ConstantBufferI1SEcvRU4AS12S1_Ev(ptr noundef nonnull align 8 dereferenceable(8) @_ZL2cb)
  // CHECK-SPIRV: [[CB_AS2:%.*]] = addrspacecast ptr addrspace(12) [[CB_CONV2]] to ptr
  // CHECK-SPIRV: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %s, ptr align 1 [[CB_AS2]], i64 8, i1 false)
  s = cb;

  // CHECK-DXIL: [[CB_CONV3:%.*]] = call noundef {{.*}} ptr addrspace(2) @_ZNK4hlsl14ConstantBufferI1SEcvRU3AS2S1_Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZL2cb)
  // CHECK-DXIL: [[CB_AS3:%.*]] = addrspacecast ptr addrspace(2) [[CB_CONV3]] to ptr
  // CHECK-DXIL: call void @llvm.memcpy.p0.p0.i32(ptr align 1 %agg.tmp, ptr align 1 [[CB_AS3]], i32 8, i1 false)
  // CHECK-DXIL: call void @_Z7takes_s1S(ptr noundef byval(%struct.S) align 1 %agg.tmp)
  // CHECK-SPIRV: [[CB_CONV3:%.*]] = call noundef {{.*}} ptr addrspace(12) @_ZNK4hlsl14ConstantBufferI1SEcvRU4AS12S1_Ev(ptr noundef nonnull align 8 dereferenceable(8) @_ZL2cb)
  // CHECK-SPIRV: [[CB_AS3:%.*]] = addrspacecast ptr addrspace(12) [[CB_CONV3]] to ptr
  // CHECK-SPIRV: call {{.*}} void @_Z7takes_s1S(ptr noundef byval(%struct.S) align 1 %agg.tmp)
  takes_s(cb);

  // CHECK: call void @_ZN4hlsl14ConstantBufferI1SEC1ERKS2_(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %agg.tmp{{[0-9]+}}, ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @_ZL2cb)
  // CHECK-DXIL: call void @_Z8takes_cbN4hlsl14ConstantBufferI1SEE(ptr noundef dead_on_return %agg.tmp{{[0-9]+}})
  // CHECK-SPIRV: call {{.*}} void @_Z8takes_cbN4hlsl14ConstantBufferI1SEE(ptr noundef dead_on_return %agg.tmp{{[0-9]+}})
  takes_cb(cb);
}
