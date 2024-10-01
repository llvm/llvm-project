// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -emit-llvm -o - %s | FileCheck %s

using handle_float_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(float)]];

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", %struct.MyStruct = type { <4 x float>, <2 x i32>, [8 x i8] }, 1, 0)

// CHECK: define void @"?fa@@YAXU__hlsl_resource_t@@uA@A@M@Z"(target("dx.TypedBuffer", float, 1, 0, 0) %a)
// CHECK: call void @"?foo1@@YAXU__hlsl_resource_t@@uA@A@M@Z"(target("dx.TypedBuffer", float, 1, 0, 0)
// CHECK: declare void @"?foo1@@YAXU__hlsl_resource_t@@uA@A@M@Z"(target("dx.TypedBuffer", float, 1, 0, 0))

void foo1(handle_float_t res);

void fa(handle_float_t a) {
    foo1(a);
}

// CHECK: define void @"?fb@@YAXU__hlsl_resource_t@@uA@A@M@Z"(target("dx.TypedBuffer", float, 1, 0, 0) %a)
void fb(handle_float_t a) {
    handle_float_t b = a;
}

// CHECK: define void @"?fc@@YAXV?$RWBuffer@T?$__vector@M$03@__clang@@@hlsl@@@Z"(ptr noundef byval(%"class.hlsl::RWBuffer") align 16 %a)
// CHECK: call void @"?foo2@@YAXV?$RWBuffer@T?$__vector@M$03@__clang@@@hlsl@@@Z"(ptr noundef byval(%"class.hlsl::RWBuffer") align 16 %agg.tmp)  
// CHECK: declare void @"?foo2@@YAXV?$RWBuffer@T?$__vector@M$03@__clang@@@hlsl@@@Z"(ptr noundef byval(%"class.hlsl::RWBuffer") align 16)
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

// CHECK: define void @"?fe@@YAXV?$StructuredBuffer@UMyStruct@@@hlsl@@@Z"(ptr noundef byval(%"class.hlsl::StructuredBuffer") align 16 %a)
// CHECK: call void @"?foo3@@YAXV?$StructuredBuffer@UMyStruct@@@hlsl@@@Z"(ptr noundef byval(%"class.hlsl::StructuredBuffer") align 16 %agg.tmp)
// CHECK: declare void @"?foo3@@YAXV?$StructuredBuffer@UMyStruct@@@hlsl@@@Z"(ptr noundef byval(%"class.hlsl::StructuredBuffer") align 16)
void foo3(StructuredBuffer<MyStruct> buf);

void fe(StructuredBuffer<MyStruct> a) {
  foo3(a);
}

void ff(StructuredBuffer<MyStruct> a) {
  StructuredBuffer<MyStruct> b = a;
}

