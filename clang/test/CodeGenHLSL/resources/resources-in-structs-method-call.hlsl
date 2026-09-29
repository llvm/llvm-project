// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -finclude-default-header -o - %s | llvm-cxxfilt | FileCheck %s

struct MyStruct {
  float f;
  RWBuffer<float> Buf;

  void Store() const {
    Buf[0] = f;
  }
};

// CHECK-DAG: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }
// CHECK-DAG: %__cblayout_CB = type <{ %__cblayout_MyStruct }>
// CHECK-DAG: %__cblayout_MyStruct = type <{ float }>
// CHECK-DAG: %struct.MyStruct = type { float, %"class.hlsl::RWBuffer" }
cbuffer CB {
  MyStruct one;
}

// CHECK-DAG: @one.Buf = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK-DAG: @CB.cb = internal global target("dx.CBuffer", %__cblayout_CB) poison
// CHECK-DAG: @one = external hidden addrspace(2) global %__cblayout_MyStruct, align 4

// $Globals constant buffer
// CHECK-DAG: @"$Globals.cb" = internal global target("dx.CBuffer", %"__cblayout_$Globals") poison
// CHECK-DAG: %"__cblayout_$Globals" = type <{ %__cblayout_MyStruct }>
// CHECK-DAG: @two.Buf = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK-DAG: @two = external hidden addrspace(2) global %struct.MyStruct, align 4
MyStruct two;

// CHECK-DAG: @Constants = internal global %"class.hlsl::ConstantBuffer" poison, align 4
// CHECK-DAG: %MyConstants = type <{ <4 x float>, <3 x i32> }>
// CHECK-DAG: %struct.MyConstants = type { <4 x float>, <3 x i32> }

struct MyConstants {
  float4 vec;
  int3 pos;
  int3 getPosition() const { return pos; }
};

ConstantBuffer<MyConstants> Constants;

// CHECK-LABEL: define internal void @main()()
[numthreads(4,1,1)]
void main() {
// CHECK: [[TMP1:%.*]] = alloca %struct.MyStruct, align 4
// CHECK-NEXT: [[TMP2:%.*]] = alloca %struct.MyStruct, align 4
// CHECK-NEXT: %pos = alloca <3 x i32>, align 4
// CHECK-NEXT: [[TMP3:%.*]] = alloca %struct.MyConstants, align 1

// Make sure we copy the struct from the constant buffer element by element to a temporary
// variable and then call the method on that.

// CHECK-NEXT: [[TMP1_F_PTR:%.*]] = getelementptr inbounds %struct.MyStruct, ptr [[TMP1]], i32 0, i32 0
// CHECK-NEXT: [[CBUFLOAD0:%.*]] = load float, ptr addrspace(2) @one, align 4
// CHECK-NEXT: store float [[CBUFLOAD0]], ptr [[TMP1_F_PTR]], align 4
// CHECK-NEXT: [[TMP1_BUF_PTR:%.*]] = getelementptr inbounds %struct.MyStruct, ptr [[TMP1]], i32 0, i32 1
// CHECK-NEXT: store ptr @one.Buf, ptr [[TMP1_BUF_PTR]], align 4
// CHECK-NEXT: call void @MyStruct::Store() const(ptr {{.*}} [[TMP1]])
  one.Store();

// CHECK-NEXT: [[TMP2_F_PTR:%.*]] = getelementptr inbounds %struct.MyStruct, ptr [[TMP2]], i32 0, i32 0
// CHECK-NEXT: [[CBUFLOAD2:%.*]] = load float, ptr addrspace(2) @two, align 4
// CHECK-NEXT: store float [[CBUFLOAD2]], ptr [[TMP2_F_PTR]], align 4
// CHECK-NEXT: [[TMP2_BUF_PTR:%.*]] = getelementptr inbounds %struct.MyStruct, ptr [[TMP2]], i32 0, i32 1
// CHECK-NEXT: store ptr @two.Buf, ptr [[TMP2_BUF_PTR]], align 4
// CHECK-NEXT: call void @MyStruct::Store() const(ptr {{.*}} [[TMP2]])
  two.Store();
  
// CHECK-NEXT: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace(2) @hlsl::ConstantBuffer<MyConstants>::operator MyConstants const AS2&() const(ptr {{.*}} @Constants)

// CHECK-NEXT: [[CB_PTR_VEC:%.*]] = getelementptr inbounds %MyConstants, ptr addrspace(2) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[TMP3_VEC_PTR:%.*]] = getelementptr inbounds %struct.MyConstants, ptr [[TMP3]], i32 0, i32 0
// CHECK-NEXT: [[CBUFLOAD3:%.*]] = load <4 x float>, ptr addrspace(2) [[CB_PTR_VEC]], align 4
// CHECK-NEXT: store <4 x float> [[CBUFLOAD3]], ptr [[TMP3_VEC_PTR]], align 4

// CHECK-NEXT: [[CB_POS_PTR:%.*]] = getelementptr inbounds %MyConstants, ptr addrspace(2) [[CB_PTR]], i32 0, i32 1
// CHECK-NEXT: [[TMP3_POS_PTR:%.*]] = getelementptr inbounds %struct.MyConstants, ptr [[TMP3]], i32 0, i32 1
// CHECK-NEXT: [[CBUFLOAD4:%.*]] = load <3 x i32>, ptr addrspace(2) [[CB_POS_PTR]], align 4
// CHECK-NEXT: store <3 x i32> [[CBUFLOAD4]], ptr [[TMP3_POS_PTR]], align 4

// CHECK-NEXT: [[POS:%.*]] = call noundef <3 x i32> @MyConstants::getPosition() const(ptr {{.*}} [[TMP3]])
// CHECK-NEXT: store <3 x i32> [[POS]], ptr %pos, align 4
  int3 pos = Constants.getPosition();
}
