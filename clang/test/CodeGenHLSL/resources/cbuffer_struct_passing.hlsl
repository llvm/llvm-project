// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:     FileCheck %s -DCONST_ADDR_SPACE=2 -DPADDING_TYPE="dx.Padding"

// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:     FileCheck %s -DCONST_ADDR_SPACE=12 -DPADDING_TYPE="spirv.Padding"

struct P {
  float3 a;
};

struct S : P {
  double b;
  float4 c;
};

struct T {
  S s;
  int arr[2];
};

// CHECK-DAG: %__cblayout_CB = type <{ %S, %T }>
// CHECK-DAG: %S = type <{ <3 x float>, target("[[PADDING_TYPE]]", 4), double, target("[[PADDING_TYPE]]", 8), <4 x float> }>
// CHECK-DAG: %T = type <{ %S, <{ [1 x <{ i32, target("[[PADDING_TYPE]]", 12) }>], i32 }> }>
// CHECK-DAG: %struct.S = type <{ %struct.P, double, <4 x float> }>
// CHECK-DAG: %struct.P = type { <3 x float> }
// CHECK-DAG: %struct.T = type { %struct.S, [2 x i32] }

cbuffer CB {
  S cbs;
  T cbt;
};
// CHECK-DAG: @cbs = external hidden addrspace([[CONST_ADDR_SPACE]]) global %S, align 1

// CHECK-LABEL: case1
// CHECK-NEXT: entry:
// CHECK-NEXT: call token @llvm.experimental.convergence.entry()
//
  // Copy S field by field into local variable in default address space.
//
// CHECK-NEXT: [[AggTemp:%.*]] = alloca %struct.S, align 1

// CHECK-NEXT: [[Ptr_a:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 0
// CHECK-NEXT: [[CBufLoad_a:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, align 4
// CHECK-NEXT: store <3 x float> [[CBufLoad_a]], ptr [[Ptr_a]], align 4

// CHECK-NEXT: [[Ptr_b:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 1
// CHECK-NEXT: [[CBufLoad_b:%.*]] = load double, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 16), align 8
// CHECK-NEXT: store double [[CBufLoad_b]], ptr [[Ptr_b]], align 8

// CHECK-NEXT: [[Ptr_c:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 2
// CHECK-NEXT: [[CBufLoad_c:%.*]] = load <4 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 32), align 4
// CHECK-NEXT: store <4 x float> [[CBufLoad_c]], ptr [[Ptr_c]], align 4
// CHECK-NEXT: ret void
void case1() {
  S local = cbs;
}

// CHECK-LABEL: case2
// CHECK-NEXT: entry:
// CHECK-NEXT: call token @llvm.experimental.convergence.entry()
//
// Copy S field by field into a temporary variable in default address space.
//
// CHECK-NEXT: [[LocalS:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[AggTemp:%.*]] = alloca %struct.S, align 1

// CHECK-NEXT: [[Ptr_a:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 0
// CHECK-NEXT: [[CBufLoad_a:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, align 4
// CHECK-NEXT: store <3 x float> [[CBufLoad_a]], ptr [[Ptr_a]], align 4

// CHECK-NEXT: [[Ptr_b:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 1
// CHECK-NEXT: [[CBufLoad_b:%.*]] = load double, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 16), align 8
// CHECK-NEXT: store double [[CBufLoad_b]], ptr [[Ptr_b]], align 8

// CHECK-NEXT: [[Ptr_c:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 2
// CHECK-NEXT: [[CBufLoad_c:%.*]] = load <4 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 32), align 4
// CHECK-NEXT: store <4 x float> [[CBufLoad_c]], ptr [[Ptr_c]], align 4
//
// The proces HLSLElementwiseCast - copy individual vector elements between the structs.
//
// CHECK-NEXT: [[Ptr_LocalS_a:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[Ptr_LocalS_b:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 1
// CHECK-NEXT: [[Ptr_LocalS_c:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 2
// CHECK-NEXT: [[Ptr_AggTemp_a:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 0, i32 0
// CHECK-NEXT: [[Ptr_AggTemp_b:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 1
// CHECK-NEXT: [[Ptr_AggTemp_c:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 2

// CHECK-NEXT: [[AggTemp_a:%.*]] = load <3 x float>, ptr [[Ptr_AggTemp_a]], align 4
// CHECK-NEXT: [[Val_a0:%.*]] = extractelement <3 x float> [[AggTemp_a]], i32 0
// CHECK-NEXT: [[Ptr_LocalS_a0:%.*]] = getelementptr <3 x float>, ptr [[Ptr_LocalS_a]], i32 0, i32 0
// CHECK-NEXT: store float [[Val_a0]], ptr [[Ptr_LocalS_a0]], align 4

// CHECK-NEXT: [[AggTemp_a:%.*]] = load <3 x float>, ptr [[Ptr_AggTemp_a]], align 4
// CHECK-NEXT: [[Val_a1:%.*]] = extractelement <3 x float> [[AggTemp_a]], i32 1
// CHECK-NEXT: [[Ptr_LocalS_a1:%.*]] = getelementptr <3 x float>, ptr [[Ptr_LocalS_a]], i32 0, i32 1
// CHECK-NEXT: store float [[Val_a1]], ptr [[Ptr_LocalS_a1]], align 4

// CHECK-NEXT: [[AggTemp_a:%.*]] = load <3 x float>, ptr [[Ptr_AggTemp_a]], align 4
// CHECK-NEXT: [[Val_a2:%.*]] = extractelement <3 x float> [[AggTemp_a]], i32 2
// CHECK-NEXT: [[Ptr_LocalS_a2:%.*]] = getelementptr <3 x float>, ptr [[Ptr_LocalS_a]], i32 0, i32 2
// CHECK-NEXT: store float [[Val_a2]], ptr [[Ptr_LocalS_a2]], align 4

// CHECK-NEXT: [[Val_b:%.*]] = load double, ptr [[Ptr_AggTemp_b]], align 8
// CHECK-NEXT: store double [[Val_b]], ptr [[Ptr_LocalS_b]], align 8

// CHECK-NEXT: [[AggTemp_c:%.*]] = load <4 x float>, ptr [[Ptr_AggTemp_c]], align 4
// CHECK-NEXT: [[Val_c0:%.*]] = extractelement <4 x float> [[AggTemp_c]], i32 0
// CHECK-NEXT: [[Ptr_LocalS_c0:%.*]] = getelementptr <4 x float>, ptr [[Ptr_LocalS_c]], i32 0, i32 0
// CHECK-NEXT: store float [[Val_c0]], ptr [[Ptr_LocalS_c0]], align 4

// CHECK-NEXT: [[AggTemp_c:%.*]] = load <4 x float>, ptr [[Ptr_AggTemp_c]], align 4
// CHECK-NEXT: [[Val_c1:%.*]] = extractelement <4 x float> [[AggTemp_c]], i32 1
// CHECK-NEXT: [[Ptr_LocalS_c1:%.*]] = getelementptr <4 x float>, ptr [[Ptr_LocalS_c]], i32 0, i32 1
// CHECK-NEXT: store float [[Val_c1]], ptr [[Ptr_LocalS_c1]], align 4

// CHECK-NEXT: [[AggTemp_c:%.*]] = load <4 x float>, ptr [[Ptr_AggTemp_c]], align 4
// CHECK-NEXT: [[Val_c2:%.*]] = extractelement <4 x float> [[AggTemp_c]], i32 2
// CHECK-NEXT: [[Ptr_LocalS_c2:%.*]] = getelementptr <4 x float>, ptr [[Ptr_LocalS_c]], i32 0, i32 2
// CHECK-NEXT: store float [[Val_c2]], ptr [[Ptr_LocalS_c2]], align 4

// CHECK-NEXT: [[AggTemp_c:%.*]] = load <4 x float>, ptr [[Ptr_AggTemp_c]], align 4
// CHECK-NEXT: [[Val_c3:%.*]] = extractelement <4 x float> [[AggTemp_c]], i32 3
// CHECK-NEXT: [[Ptr_LocalS_c3:%.*]] = getelementptr <4 x float>, ptr [[Ptr_LocalS_c]], i32 0, i32 3
// CHECK-NEXT: store float [[Val_c3]], ptr [[Ptr_LocalS_c3]], align 4

// CHECK-NEXT: ret void
void case2() {
  S LocalS = (S)cbs;
}

// CHECK-LABEL: case3
// CHECK-NEXT: entry:
// CHECK-NEXT: call token @llvm.experimental.convergence.entry()
void case3() {

// CHECK-NEXT: [[LocalT:%.*]] = alloca %struct.T, align 1
// CHECK-NEXT: [[LocalTCopy:%.*]] = alloca %struct.T, align 1
// CHECK-NEXT: [[LocalS:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[LocalSCopy:%.*]] = alloca %struct.S, align 1

// Check that constant to default address space copies the struct field by field
//
// CHECK-NEXT: [[Ptr_s:%.*]] = getelementptr inbounds %struct.T, ptr [[LocalT]], i32 0, i32 0
// CHECK-NEXT: [[Ptr_a:%.*]] = getelementptr inbounds %struct.S, ptr [[Ptr_s]], i32 0, i32 0
// CHECK-NEXT: [[CbufLoad_a:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) @cbt, align 4
// CHECK-NEXT: store <3 x float> [[CbufLoad_a]], ptr [[Ptr_a]], align 4

// CHECK-NEXT: [[Ptr_b:%.*]] = getelementptr inbounds %struct.S, ptr [[Ptr_s]], i32 0, i32 1
// CHECK-NEXT: [[CbufLoad_c:%.*]] = load double, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbt, {{i32|i64}} 16), align 8
// CHECK-NEXT: store double [[CbufLoad_c]], ptr [[Ptr_b]], align 8

// CHECK-NEXT: [[Ptr_c:%.*]] = getelementptr inbounds %struct.S, ptr [[Ptr_s]], i32 0, i32 2
// CHECK-NEXT: [[CbufLoad_b:%.*]] = load <4 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbt, {{i32|i64}} 32), align 4
// CHECK-NEXT: store <4 x float> [[CbufLoad_b]], ptr [[Ptr_c]], align 4
  
// CHECK-NEXT: [[Ptr_arr:%.*]] = getelementptr inbounds %struct.T, ptr [[LocalT]], i32 0, i32 1
// CHECK-NEXT: [[Ptr_arr0:%.*]] = getelementptr inbounds [2 x i32], ptr [[Ptr_arr]], i32 0, i32 0
// CHECK-NEXT: [[CbufLoad_arr0:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbt, {{i32|i64}} 48), align 4
// CHECK-NEXT: store i32 [[CbufLoad_arr0]], ptr [[Ptr_arr0]], align 4

// CHECK-NEXT: [[Ptr_arr1:%.*]] = getelementptr inbounds [2 x i32], ptr [[Ptr_arr]], i32 0, i32 1
// CHECK-NEXT: [[CbufLoad_arr1:%.*]] = load i32, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbt, {{i32|i64}} 64), align 4
// CHECK-NEXT: store i32 [[CbufLoad_arr1]], ptr [[Ptr_arr1]], align 4
  T localT = cbt;

// Check that default to default address space copy uses memcpy
//
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.{{i32|i64}}(ptr align 1 [[LocalTCopy]], ptr align 1 [[LocalT]], {{i32|i64}} 44, i1 false)
  T localTCopy = localT;

// Check that constant to default address space copies the struct field by field
//
// CHECK-NEXT: [[Ptr_a1:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 0
// CHECK-NEXT: [[CbufLoad_a1:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) @cbt, align 4
// CHECK-NEXT: store <3 x float> [[CbufLoad_a1]], ptr [[Ptr_a1]], align 4
// CHECK-NEXT: [[Ptr_b1:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 1
// CHECK-NEXT: [[CbufLoad_b1:%.*]] = load double, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbt, {{i32|i64}} 16), align 8
// CHECK-NEXT: store double [[CbufLoad_b1]], ptr [[Ptr_b1]], align 8
// CHECK-NEXT: [[Ptr_c1:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 2
// CHECK-NEXT: [[CbufLoad_c1:%.*]] = load <4 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbt, {{i32|i64}} 32), align 4
// CHECK-NEXT: store <4 x float> [[CbufLoad_c1]], ptr [[Ptr_c1]], align 4
  S localS = cbt.s;

// Check that default to default address space copy uses memcpy
//
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.{{i32|i64}}(ptr align 1 [[LocalSCopy]], ptr align 1 [[LocalS]], {{i32|i64}} 36, i1 false)
  S localSCopy = localS;

// CHECK-NEXT: ret void
}

// CHECK-LABEL: case4
// CHECK-NEXT: entry:
// CHECK-NEXT: call token @llvm.experimental.convergence.entry()
void case4() {

// CHECK-NEXT: [[LocalP1:%.*]] = alloca %struct.P, align 1
// CHECK-NEXT: [[Tmp0:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[LocalP2:%.*]] = alloca %struct.P, align 1
// CHECK-NEXT: [[Tmp1:%.*]] = alloca %struct.P, align 1
// CHECK-NEXT: [[Tmp2:%.*]] = alloca %struct.S, align 1

// CHECK-NEXT: [[Tmp0Ptr_a1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp0]], i32 0, i32 0
// CHECK-NEXT: [[CbufLoad_a1:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, align 4
// CHECK-NEXT: store <3 x float> [[CbufLoad_a1]], ptr [[Tmp0Ptr_a1]], align 4

// CHECK-NEXT: [[Tmp0Ptr_b1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp0]], i32 0, i32 1
// CHECK-NEXT: [[CbufLoad_b1:%.*]] = load double, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 16), align 8
// CHECK-NEXT: store double [[CbufLoad_b1]], ptr [[Tmp0Ptr_b1]], align 8

// CHECK-NEXT: [[Tmp0Ptr_c1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp0]], i32 0, i32 2
// CHECK-NEXT: [[CbufLoad_c1:%.*]] = load <4 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 32), align 4
// CHECK-NEXT: store <4 x float> [[CbufLoad_c1]], ptr [[Tmp0Ptr_c1]], align 4

// CHECK-NEXT: call void @llvm.memcpy.p0.p0.{{i32|i64}}(ptr align 1 [[LocalP1]], ptr align 1 [[Tmp0]], {{i32|i64}} 12, i1 false)

  // Derived to base conversion in initialization. Size of S in memory layout is 36 bytes and
  // size of P is 12 bytes. The memcpy should only copy the 12 bytes of P.
  P LocalP1 = cbs;

// CHECK-NEXT: [[Tmp2Ptr_a1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp2]], i32 0, i32 0
// CHECK-NEXT: [[CbufLoad_a1:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, align 4
// CHECK-NEXT: store <3 x float> [[CbufLoad_a1]], ptr [[Tmp2Ptr_a1]], align 4
// CHECK-NEXT: [[Tmp2Ptr_b1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp2]], i32 0, i32 1
// CHECK-NEXT: [[CbufLoad_b1:%.*]] = load double, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 16), align 8
// CHECK-NEXT: store double [[CbufLoad_b1]], ptr [[Tmp2Ptr_b1]], align 8
// CHECK-NEXT: [[Tmp2Ptr_c1:%.*]] = getelementptr inbounds %struct.S, ptr [[Tmp2]], i32 0, i32 2
// CHECK-NEXT: [[CbufLoad_c1:%.*]] = load <4 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 32), align 4
// CHECK-NEXT: store <4 x float> [[CbufLoad_c1]], ptr [[Tmp2Ptr_c1]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.{{i32|i64}}(ptr align 1 [[LocalP2]], ptr align 1 [[Tmp2]], {{i32|i64}} 12, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.{{i32|i64}}(ptr align 1 [[Tmp1]], ptr align 1 [[LocalP2]], {{i32|i64}} 12, i1 false)

  // Derived to base conversion in assignment. Size of S in memory layout is 36 bytes and
  // size of P is 12 bytes. The memcpy should only copy the 12 bytes of P.
  P LocalP2;
  LocalP2 = cbs;

// CHECK-NEXT: ret void
}
