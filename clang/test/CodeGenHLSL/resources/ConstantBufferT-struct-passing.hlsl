// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:     llvm-cxxfilt | FileCheck %s -DCONST_ADDR_SPACE=2 -DPADDING_TYPE="dx.Padding" -check-prefixes=CHECK,CHECK-DXIL

// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:     llvm-cxxfilt | FileCheck %s -DCONST_ADDR_SPACE=12 -DPADDING_TYPE="spirv.Padding" -check-prefixes=CHECK,CHECK-SPIRV

struct P {
  float a;
};

struct S : P {
  double2 b;
};

struct T {
  P p;
  int arr[2];
};

ConstantBuffer<S> CBS;
ConstantBuffer<T> CBT;

// CHECK-DXIL: %"class.hlsl::ConstantBuffer" = type { target("dx.CBuffer", %S) }
// CHECK-SPIRV: %"class.hlsl::ConstantBuffer" = type { target("spirv.VulkanBuffer", %S, 2, 0) }
// CHECK: %S = type <{ float, target("[[PADDING_TYPE]]", 12), <2 x double> }>

// CHECK-DXIL: %"class.hlsl::ConstantBuffer.0" = type { target("dx.CBuffer", %T) }
// CHECK-SPIRV: %"class.hlsl::ConstantBuffer.0" = type { target("spirv.VulkanBuffer", %T, 2, 0) }
// CHECK: %T = type <{ %P, target("[[PADDING_TYPE]]", 12), <{ [1 x <{ i32, target("[[PADDING_TYPE]]", 12) }>], i32 }> }>
// CHECK: %P = type <{ float }>

// CHECK: %struct.S = type <{ %struct.P, <2 x double> }>
// CHECK: %struct.P = type { float }
// CHECK: %struct.T = type { %struct.P, [2 x i32] }

// CHECK: @CBS = internal global %"class.hlsl::ConstantBuffer" poison, align {{(4|8)}}
// CHECK: @CBT = internal global %"class.hlsl::ConstantBuffer.0" poison, align {{(4|8)}}

void useS(S s) {}
void useP(P p) {}
void useT(T t) {}

// CHECK-LABEL: case1
void case1() {
// CHECK: %s = alloca %struct.S, align 1
// CHECK: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<S>::operator S const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CBS)

// s.a
// CHECK-NEXT: [[CB_A_PTR:%.*]] = getelementptr inbounds %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[S_A_PTR:%.*]] = getelementptr inbounds %struct.S, ptr %s, i32 0, i32 0
// CHECK-NEXT: [[CBUFLOAD:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_A_PTR]], align 4
// CHECK-NEXT: store float [[CBUFLOAD]], ptr [[S_A_PTR]], align 4

// s.b
// CHECK-NEXT: [[CB_B_PTR:%.*]] = getelementptr inbounds %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 2
// CHECK-NEXT: [[S_B_PTR:%.*]] = getelementptr inbounds %struct.S, ptr %s, i32 0, i32 1
// CHECK-NEXT: [[CBUFLOAD2:%.*]] = load <2 x double>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_B_PTR]], align 8
// CHECK-NEXT: store <2 x double> [[CBUFLOAD2]], ptr [[S_B_PTR]], align 8
  S s = CBS;
}

// CHECK-LABEL: case2
void case2() {
// CHECK: %s = alloca %struct.S, align 1
// CHECK: [[TMP:%.*]] = alloca %struct.S, align 1
// CHECK: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<S>::operator S const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CBS)

// s.a
// CHECK-NEXT: [[CB_A_PTR:%.*]] = getelementptr inbounds %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[S_A_PTR:%.*]] = getelementptr inbounds %struct.S, ptr %s, i32 0, i32 0
// CHECK-NEXT: [[CBUFLOAD:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_A_PTR]], align 4
// CHECK-NEXT: store float [[CBUFLOAD]], ptr [[S_A_PTR]], align 4

// s.b
// CHECK-NEXT: [[CB_B_PTR:%.*]] = getelementptr inbounds %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 2
// CHECK-NEXT: [[S_B_PTR:%.*]] = getelementptr inbounds %struct.S, ptr %s, i32 0, i32 1
// CHECK-NEXT: [[CBUFLOAD2:%.*]] = load <2 x double>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_B_PTR]], align 8
// CHECK-NEXT: store <2 x double> [[CBUFLOAD2]], ptr [[S_B_PTR]], align 8

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i{{(32|64)}}(ptr align 1 [[TMP]], ptr align 1 %s, i{{(32|64)}} 20, i1 false)
  S s;
  s = CBS;
}

// CHECK-LABEL: case3
void case3() {
// CHECK: [[TMP:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<S>::operator S const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CBS)

// tmp.a
// CHECK-NEXT: [[CB_A_PTR:%.*]] = getelementptr inbounds %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[TMP_A_PTR:%.*]] = getelementptr inbounds %struct.S, ptr [[TMP]], i32 0, i32 0
// CHECK-NEXT: [[CBUFLOAD:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_A_PTR]], align 4
// CHECK-NEXT: store float [[CBUFLOAD]], ptr [[TMP_A_PTR]], align 4
  
// tmp.b
// CHECK-NEXT: [[CB_B_PTR:%.*]] = getelementptr inbounds %S, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 2
// CHECK-NEXT: [[TMP_B_PTR:%.*]] = getelementptr inbounds %struct.S, ptr [[TMP]], i32 0, i32 1
// CHECK-NEXT: [[CBUFLOAD2:%.*]] = load <2 x double>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_B_PTR]], align 8
// CHECK-NEXT: store <2 x double> [[CBUFLOAD2]], ptr [[TMP_B_PTR]], align 8

// CHECK-NEXT: call {{.*}}void @useS(S)(ptr noundef align 1 dead_on_return [[TMP]])
  useS(CBS);
}

// CHECK-LABEL: case4
void case4() {

// CHECK: %t = alloca %struct.T, align 1
// CHECK: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<T>::operator T const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CBT)

// t.p
// CHECK-NEXT: [[CB_P_PTR:%.*]] = getelementptr inbounds %T, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[TMP_P_PTR:%.*]] = getelementptr inbounds %struct.T, ptr %t, i32 0, i32 0

// t.p.a
// CHECK-NEXT: [[CB_P_A_PTR:%.*]] = getelementptr inbounds %P, ptr addrspace([[CONST_ADDR_SPACE]]) %1, i32 0, i32 0
// CHECK-NEXT: [[TMP_P_A_PTR:%.*]] = getelementptr inbounds %struct.P, ptr %2, i32 0, i32 0
// CHECK-NEXT: [[CBUF_LOAD1:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_P_A_PTR]], align 4
// CHECK-NEXT: store float [[CBUF_LOAD1]], ptr [[TMP_P_A_PTR]], align 4

// t.arr
// CHECK-NEXT: [[CB_ARR_PTR:%.*]] = getelementptr inbounds %T, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 2
// CHECK-NEXT: [[TMP_ARR_PTR:%.*]] = getelementptr inbounds %struct.T, ptr %t, i32 0, i32 1

// t.arr[0]
// CHECK-NEXT: [[CB_ARR_0_PTR:%.*]] = getelementptr inbounds <{ [1 x <{ i32, target("[[PADDING_TYPE]]", 12) }>], i32 }>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_PTR]], i32 0, i32 0, i32 0, i32 0
// CHECK-NEXT: [[TMP_ARR_0_PTR:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP_ARR_PTR]], i32 0, i32 0
// CHECK-NEXT: [[CBUF_LOAD2:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_0_PTR]], align 4
// CHECK-NEXT: store i32 [[CBUF_LOAD2]], ptr [[TMP_ARR_0_PTR]], align 4

// t.arr[1]
// CHECK-NEXT: [[CB_ARR_1_PTR:%.*]] = getelementptr inbounds <{ [1 x <{ i32, target("[[PADDING_TYPE]]", 12) }>], i32 }>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_PTR]], i32 0, i32 1
// CHECK-NEXT: [[TMP_ARR_1_PTR:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP_ARR_PTR]], i32 0, i32 1
// CHECK-NEXT: [[CBUF_LOAD3:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_1_PTR]], align 4
// CHECK-NEXT: store i32 [[CBUF_LOAD3]], ptr [[TMP_ARR_1_PTR]], align 4

  T t = CBT;
}

// CHECK-LABEL: case5
void case5() {
// CHECK: %t = alloca %struct.T, align 1
// CHECK: [[TMP:%.*]] = alloca %struct.T, align 1
// CHECK: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<T>::operator T const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CBT)

// t.p
// CHECK-NEXT: [[CB_P_PTR:%.*]] = getelementptr inbounds %T, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[TMP_P_PTR:%.*]] = getelementptr inbounds %struct.T, ptr %t, i32 0, i32 0

// t.p.a
// CHECK-NEXT: [[CB_P_A_PTR:%.*]] = getelementptr inbounds %P, ptr addrspace([[CONST_ADDR_SPACE]]) %1, i32 0, i32 0
// CHECK-NEXT: [[TMP_P_A_PTR:%.*]] = getelementptr inbounds %struct.P, ptr %2, i32 0, i32 0
// CHECK-NEXT: [[CBUF_LOAD1:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_P_A_PTR]], align 4
// CHECK-NEXT: store float [[CBUF_LOAD1]], ptr [[TMP_P_A_PTR]], align 4

// t.arr
// CHECK-NEXT: [[CB_ARR_PTR:%.*]] = getelementptr inbounds %T, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 2
// CHECK-NEXT: [[TMP_ARR_PTR:%.*]] = getelementptr inbounds %struct.T, ptr %t, i32 0, i32 1

// t.arr[0]
// CHECK-NEXT: [[CB_ARR_0_PTR:%.*]] = getelementptr inbounds <{ [1 x <{ i32, target("[[PADDING_TYPE]]", 12) }>], i32 }>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_PTR]], i32 0, i32 0, i32 0, i32 0
// CHECK-NEXT: [[TMP_ARR_0_PTR:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP_ARR_PTR]], i32 0, i32 0
// CHECK-NEXT: [[CBUF_LOAD2:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_0_PTR]], align 4
// CHECK-NEXT: store i32 [[CBUF_LOAD2]], ptr [[TMP_ARR_0_PTR]], align 4

// t.arr[1]
// CHECK-NEXT: [[CB_ARR_1_PTR:%.*]] = getelementptr inbounds <{ [1 x <{ i32, target("[[PADDING_TYPE]]", 12) }>], i32 }>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_PTR]], i32 0, i32 1
// CHECK-NEXT: [[TMP_ARR_1_PTR:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP_ARR_PTR]], i32 0, i32 1
// CHECK-NEXT: [[CBUF_LOAD3:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_1_PTR]], align 4
// CHECK-NEXT: store i32 [[CBUF_LOAD3]], ptr [[TMP_ARR_1_PTR]], align 4

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i{{(32|64)}}(ptr align 1 [[TMP]], ptr align 1 %t, i{{(32|64)}} 12, i1 false)

  T t;
  t = CBT;
}

// CHECK-LABEL: case6
void case6() {
// CHECK: [[TMP:%.*]] = alloca %struct.T, align 1
// CHECK: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<T>::operator T const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CBT)

// t.p
// CHECK-NEXT: [[CB_P_PTR:%.*]] = getelementptr inbounds %T, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[TMP_P_PTR:%.*]] = getelementptr inbounds %struct.T, ptr [[TMP]], i32 0, i32 0

// t.p.a
// CHECK-NEXT: [[CB_P_A_PTR:%.*]] = getelementptr inbounds %P, ptr addrspace([[CONST_ADDR_SPACE]]) %1, i32 0, i32 0
// CHECK-NEXT: [[TMP_P_A_PTR:%.*]] = getelementptr inbounds %struct.P, ptr %2, i32 0, i32 0
// CHECK-NEXT: [[CBUF_LOAD1:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_P_A_PTR]], align 4
// CHECK-NEXT: store float [[CBUF_LOAD1]], ptr [[TMP_P_A_PTR]], align 4

// t.arr
// CHECK-NEXT: [[CB_ARR_PTR:%.*]] = getelementptr inbounds %T, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 2
// CHECK-NEXT: [[TMP_ARR_PTR:%.*]] = getelementptr inbounds %struct.T, ptr [[TMP]], i32 0, i32 1

// t.arr[0]
// CHECK-NEXT: [[CB_ARR_0_PTR:%.*]] = getelementptr inbounds <{ [1 x <{ i32, target("[[PADDING_TYPE]]", 12) }>], i32 }>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_PTR]], i32 0, i32 0, i32 0, i32 0
// CHECK-NEXT: [[TMP_ARR_0_PTR:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP_ARR_PTR]], i32 0, i32 0
// CHECK-NEXT: [[CBUF_LOAD2:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_0_PTR]], align 4
// CHECK-NEXT: store i32 [[CBUF_LOAD2]], ptr [[TMP_ARR_0_PTR]], align 4

// t.arr[1]
// CHECK-NEXT: [[CB_ARR_1_PTR:%.*]] = getelementptr inbounds <{ [1 x <{ i32, target("[[PADDING_TYPE]]", 12) }>], i32 }>, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_PTR]], i32 0, i32 1
// CHECK-NEXT: [[TMP_ARR_1_PTR:%.*]] = getelementptr inbounds [2 x i32], ptr [[TMP_ARR_PTR]], i32 0, i32 1
// CHECK-NEXT: [[CBUF_LOAD3:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_ARR_1_PTR]], align 4
// CHECK-NEXT: store i32 [[CBUF_LOAD3]], ptr [[TMP_ARR_1_PTR]], align 4
  useT(CBT);
}

// CHECK-LABEL: case7
void case7() {
// CHECK: %p = alloca %struct.P, align 1
// CHECK: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<T>::operator T const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CBT)  

// CHECK-NEXT: [[CB_P_PTR:%.*]] = getelementptr inbounds nuw %T, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[CB_P_A_PTR:%.*]] = getelementptr inbounds %P, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_P_PTR]], i32 0, i32 0
// CHECK-NEXT: [[P_A_PTR:%.*]] = getelementptr inbounds %struct.P, ptr %p, i32 0, i32 0
// CHECK-NEXT: [[CBUF_LOAD:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_P_A_PTR]], align 4
// CHECK-NEXT: store float [[CBUF_LOAD]], ptr [[P_A_PTR]], align 4
  P p = CBT.p;
}

// CHECK-LABEL: case8
void case8() {
// CHECK: %p = alloca %struct.P, align 1
// CHECK: [[TMP:%.*]] = alloca %struct.P, align 1
// CHECK: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<T>::operator T const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CBT)  

// CHECK-NEXT: [[CB_P_PTR:%.*]] = getelementptr inbounds nuw %T, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[CB_P_A_PTR:%.*]] = getelementptr inbounds %P, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_P_PTR]], i32 0, i32 0
// CHECK-NEXT: [[P_A_PTR:%.*]] = getelementptr inbounds %struct.P, ptr %p, i32 0, i32 0
// CHECK-NEXT: [[CBUF_LOAD:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_P_A_PTR]], align 4
// CHECK-NEXT: store float [[CBUF_LOAD]], ptr [[P_A_PTR]], align 4

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i{{(32|64)}}(ptr align 1 [[TMP]], ptr align 1 %p, i{{(32|64)}} 4, i1 false)
  P p;
  p = CBT.p;
}

// CHECK-LABEL: case9
void case9() {
// CHECK: [[TMP:%.*]] = alloca %struct.P, align 1
// CHECK: [[CB_PTR:%.*]] = call {{.*}} ptr addrspace([[CONST_ADDR_SPACE]]) @hlsl::ConstantBuffer<T>::operator T const AS[[CONST_ADDR_SPACE]]&() const(ptr {{.*}} @CBT)  

// CHECK-NEXT: [[CB_P_PTR:%.*]] = getelementptr inbounds nuw %T, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_PTR]], i32 0, i32 0
// CHECK-NEXT: [[CB_P_A_PTR:%.*]] = getelementptr inbounds %P, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_P_PTR]], i32 0, i32 0
// CHECK-NEXT: [[P_A_PTR:%.*]] = getelementptr inbounds %struct.P, ptr [[TMP]], i32 0, i32 0
// CHECK-NEXT: [[CBUF_LOAD:%.*]] = load float, ptr addrspace([[CONST_ADDR_SPACE]]) [[CB_P_A_PTR]], align 4
// CHECK-NEXT: store float [[CBUF_LOAD]], ptr [[P_A_PTR]], align 4

// CHECK-NEXT: call {{.*}}void @useP(P)(ptr noundef align 1 dead_on_return [[TMP]])
  useP(CBT.p);
}
