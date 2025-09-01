// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -emit-llvm -o - %s | FileCheck %s

using handle_float_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(float)]];

struct CustomResource {
  handle_float_t h;
};

// CHECK: %struct.CustomResource = type { target("dx.TypedBuffer", float, 1, 0, 0) }
// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", %struct.MyStruct, 0, 0)
// CHECK: %struct.MyStruct = type <{ <4 x float>, <2 x i32> }>

// CHECK: define hidden void @_Z2fa14CustomResource(ptr noundef byval(%struct.CustomResource) align 1 %a)
// CHECK: call void @_Z4foo114CustomResource(ptr noundef byval(%struct.CustomResource) align 1 %agg.tmp)
// CHECK: declare hidden void @_Z4foo114CustomResource(ptr noundef byval(%struct.CustomResource) align 1)

void foo1(CustomResource res);

void fa(CustomResource a) {
    foo1(a);
}

// CHECK: define hidden void @_Z2fb14CustomResource(ptr noundef byval(%struct.CustomResource) align 1 %a)
void fb(CustomResource a) {
    CustomResource b = a;
}

// CHECK: define hidden void @_Z2fcN4hlsl8RWBufferIDv4_fEE(ptr noundef byval(%"class.hlsl::RWBuffer") align 4 %a)
// CHECK: call void @_Z4foo2N4hlsl8RWBufferIDv4_fEE(ptr noundef byval(%"class.hlsl::RWBuffer") align 4 %agg.tmp)
// CHECK: declare hidden void @_Z4foo2N4hlsl8RWBufferIDv4_fEE(ptr noundef byval(%"class.hlsl::RWBuffer") align 4)
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

// CHECK: define hidden void @_Z2feN4hlsl16StructuredBufferI8MyStructEE(ptr noundef byval(%"class.hlsl::StructuredBuffer") align 4 %a)
// CHECK: call void @_Z4foo3N4hlsl16StructuredBufferI8MyStructEE(ptr noundef byval(%"class.hlsl::StructuredBuffer") align 4 %agg.tmp)
// CHECK: declare hidden void @_Z4foo3N4hlsl16StructuredBufferI8MyStructEE(ptr noundef byval(%"class.hlsl::StructuredBuffer") align 4)
void foo3(StructuredBuffer<MyStruct> buf);

void fe(StructuredBuffer<MyStruct> a) {
  foo3(a);
}

void ff(StructuredBuffer<MyStruct> a) {
  StructuredBuffer<MyStruct> b = a;
}
