// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -emit-llvm -o - %s | FileCheck %s

using handle_float_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(float)]];

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", %struct.MyStruct, 0, 0)
// CHECK: %struct.MyStruct = type { <4 x float>, <2 x i32>, [8 x i8] }

// CHECK: define void @_Z2faU9_Res_u_CTfu17__hlsl_resource_t(target("dx.TypedBuffer", float, 1, 0, 0) %a)
// CHECK: call void @_Z4foo1U9_Res_u_CTfu17__hlsl_resource_t(target("dx.TypedBuffer", float, 1, 0, 0) %0)
// CHECK: declare void @_Z4foo1U9_Res_u_CTfu17__hlsl_resource_t(target("dx.TypedBuffer", float, 1, 0, 0))

void foo1(handle_float_t res);

void fa(handle_float_t a) {
    foo1(a);
}

// CHECK: define void @_Z2fbU9_Res_u_CTfu17__hlsl_resource_t(target("dx.TypedBuffer", float, 1, 0, 0) %a)
void fb(handle_float_t a) {
    handle_float_t b = a;
}

// CHECK: define void @_Z2fcN4hlsl8RWBufferIDv4_fEE(ptr noundef byval(%"class.hlsl::RWBuffer") align 4 %a)
// CHECK: call void @_Z4foo2N4hlsl8RWBufferIDv4_fEE(ptr noundef byval(%"class.hlsl::RWBuffer") align 4 %agg.tmp)
// CHECK: declare void @_Z4foo2N4hlsl8RWBufferIDv4_fEE(ptr noundef byval(%"class.hlsl::RWBuffer") align 4)
void foo2(RWBuffer<float4> buf);

void fc(RWBuffer<float4> a) {
  foo2(a);
}

void fd(RWBuffer<float4> a) {
  RWBuffer<float4> b = a;
}

struct MyStruct {
  float4 f;
  int2 i;
};

// CHECK: define void @_Z2feN4hlsl16StructuredBufferI8MyStructEE(ptr noundef byval(%"class.hlsl::StructuredBuffer") align 4 %a)
// CHECK: call void @_Z4foo3N4hlsl16StructuredBufferI8MyStructEE(ptr noundef byval(%"class.hlsl::StructuredBuffer") align 4 %agg.tmp)
// CHECK: declare void @_Z4foo3N4hlsl16StructuredBufferI8MyStructEE(ptr noundef byval(%"class.hlsl::StructuredBuffer") align 4)
void foo3(StructuredBuffer<MyStruct> buf);

void fe(StructuredBuffer<MyStruct> a) {
  foo3(a);
}

void ff(StructuredBuffer<MyStruct> a) {
  StructuredBuffer<MyStruct> b = a;
}
