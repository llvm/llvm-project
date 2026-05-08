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
// CHECK-DXIL: [[CB_CONV:%.*]] = call noundef {{.*}} ptr addrspace(2) @_ZNK4hlsl14ConstantBufferI1SEcvRU3AS2KS1_Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZL2cb)
// CHECK-DXIL: [[GEP_A:%.*]] = getelementptr inbounds nuw %S, ptr addrspace(2) [[CB_CONV]], i32 0, i32 0
// CHECK-DXIL: [[LOAD_A:%.*]] = load float, ptr addrspace(2) [[GEP_A]], align 4

// CHECK-SPIRV: [[CB_CONV:%.*]] = call noundef {{.*}} ptr addrspace(12) @_ZNK4hlsl14ConstantBufferI1SEcvRU4AS12KS1_Ev(ptr noundef nonnull align 8 dereferenceable(8) @_ZL2cb)
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
  // CHECK-DXIL: [[CB_CONV:%.*]] = call noundef {{.*}} ptr addrspace(2) @_ZNK4hlsl14ConstantBufferI6NestedEcvRU3AS2KS1_Ev(ptr noundef nonnull align 4 dereferenceable(4) [[TMP_CB]])
  // CHECK-DXIL: [[GEP_S:%.*]] = getelementptr inbounds nuw %Nested, ptr addrspace(2) [[CB_CONV]], i32 0, i32 0
  // CHECK-DXIL: [[GEP_A2:%.*]] = getelementptr inbounds nuw %S, ptr addrspace(2) [[GEP_S]], i32 0, i32 0
  // CHECK-DXIL: [[LOAD_A2:%.*]] = load float, ptr addrspace(2) [[GEP_A2]], align 4

  // CHECK-SPIRV: [[TMP_CB:%.*]] = alloca %"class.hlsl::ConstantBuffer.0", align 8
  // CHECK-SPIRV: call void @_ZN4hlsl14ConstantBufferI6NestedE27__createFromImplicitBindingEjjijPKc(ptr dead_on_unwind writable sret(%"class.hlsl::ConstantBuffer.0") align 8 [[TMP_CB]], i32 noundef 1, i32 noundef 0, i32 noundef 2, i32 noundef 1, ptr noundef @cb_nested.str)
  // CHECK-SPIRV: [[CB_CONV:%.*]] = call noundef {{.*}} ptr addrspace(12) @_ZNK4hlsl14ConstantBufferI6NestedEcvRU4AS12KS1_Ev(ptr noundef nonnull align 8 dereferenceable(8) [[TMP_CB]])
  // CHECK-SPIRV: [[GEP_S:%.*]] = getelementptr inbounds nuw %Nested, ptr addrspace(12) [[CB_CONV]], i32 0, i32 0
  // CHECK-SPIRV: [[GEP_A2:%.*]] = getelementptr inbounds nuw %S, ptr addrspace(12) [[GEP_S]], i32 0, i32 0
  // CHECK-SPIRV: [[LOAD_A2:%.*]] = load float, ptr addrspace(12) [[GEP_A2]], align 4

  // CHECK: store float [[LOAD_A2]], ptr %f2, align 4
  float f2 = cb_nested[1].s.a;
}

void takes_cb(ConstantBuffer<S> c) {}

[numthreads(1,1,1)]
void test_params() {
  // CHECK-LABEL: define {{.*}} void @_Z11test_paramsv()
  // CHECK: call void @_ZN4hlsl14ConstantBufferI1SEC1ERKS2_(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %agg.tmp, ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @_ZL2cb)
  // CHECK-DXIL: call void @_Z8takes_cbN4hlsl14ConstantBufferI1SEE(ptr noundef dead_on_return %agg.tmp)
  // CHECK-SPIRV: call {{.*}} void @_Z8takes_cbN4hlsl14ConstantBufferI1SEE(ptr noundef dead_on_return %agg.tmp)
  takes_cb(cb);
}
