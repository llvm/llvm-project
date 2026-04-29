// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:     FileCheck %s -DCONST_ADDR_SPACE=2 -DPADDING_TYPE="dx.Padding"

// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | \
// RUN:     FileCheck %s -DCONST_ADDR_SPACE=12 -DPADDING_TYPE="spirv.Padding" --check-prefixes=CHECK,SPIRV

struct P {
  float3 a;
};

struct S : P {
  float4 b;
};

// CHECK-DAG: %S = type <{ <3 x float>, target("[[PADDING_TYPE]]", 4), <4 x float> }>
// CHECK-DAG: %struct.P = type { <3 x float> }
// CHECK-DAG: %struct.S = type { %struct.P, <4 x float> }

cbuffer CB {
  S cbs;
};
// CHECK-DAG: @cbs = external hidden addrspace([[CONST_ADDR_SPACE]]) global %S, align 1

// CHECK-LABEL: case1
// CHECK-NEXT: entry:
// SPIRV-NEXT: call token @llvm.experimental.convergence.entry()
//
  // Copy S field by field into local variable in default address space.
//
// CHECK-NEXT: [[LocalS:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[PtrA:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 0
// CHECK-NEXT: [[CBufLoad1:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, align 4
// CHECK-NEXT: store <3 x float> [[CBufLoad1]], ptr [[PtrA]], align 4
// CHECK-NEXT: [[PtrB:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 1
// CHECK-NEXT: [[CBufLoad2:%.*]] = load <4 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 16), align 4
// CHECK-NEXT: store <4 x float> [[CBufLoad2]], ptr [[PtrB]], align 4
// CHECK-NEXT: ret void
void case1() {
  S local = cbs;
}

// CHECK-LABEL: case2
// CHECK-NEXT: entry:
// SPIRV-NEXT: call token @llvm.experimental.convergence.entry()
//
// Copy S field by field into a temporary variable in default address space.
//
// CHECK-NEXT: [[LocalS:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[AggTemp:%.*]] = alloca %struct.S, align 1
// CHECK-NEXT: [[PtrA:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 0
// CHECK-NEXT: [[CBufLoad1:%.*]] = load <3 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, align 4
// CHECK-NEXT: store <3 x float> [[CBufLoad1]], ptr [[PtrA]], align 4
// CHECK-NEXT: [[PtrB:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 1
// CHECK-NEXT: [[CBufLoad2:%.*]] = load <4 x float>, ptr addrspace([[CONST_ADDR_SPACE]]) getelementptr inbounds nuw (i8, ptr addrspace([[CONST_ADDR_SPACE]]) @cbs, {{i32|i64}} 16), align 4
// CHECK-NEXT: store <4 x float> [[CBufLoad2]], ptr [[PtrB]], align 4
//
// The proces HLSLElementwiseCast - copy individual vector elements between the structs.
//
// CHECK-NEXT: [[VecGep1:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 0
// CHECK-NEXT: [[VecGep2:%.*]] = getelementptr inbounds %struct.S, ptr [[LocalS]], i32 0, i32 1
// CHECK-NEXT: [[VecGep3:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 0
// CHECK-NEXT: [[VecGep4:%.*]] = getelementptr inbounds %struct.S, ptr [[AggTemp]], i32 0, i32 1

// CHECK-NEXT: [[VecA1:%.*]] = load <3 x float>, ptr [[VecGep3]], align 4
// CHECK-NEXT: [[Val1:%.*]] = extractelement <3 x float> [[VecA1]], i32 0
// CHECK-NEXT: [[VecA1Ptr:%.*]] = getelementptr <3 x float>, ptr [[VecGep1]], i32 0, i32 0
// CHECK-NEXT: store float [[Val1]], ptr [[VecA1Ptr]], align 4

// CHECK-NEXT: [[VecA2:%.*]] = load <3 x float>, ptr [[VecGep3]], align 4
// CHECK-NEXT: [[Val2:%.*]] = extractelement <3 x float> [[VecA2]], i32 1
// CHECK-NEXT: [[VecA2Ptr:%.*]] = getelementptr <3 x float>, ptr [[VecGep1]], i32 0, i32 1
// CHECK-NEXT: store float [[Val2]], ptr [[VecA2Ptr]], align 4

// CHECK-NEXT: [[VecA3:%.*]] = load <3 x float>, ptr [[VecGep3]], align 4
// CHECK-NEXT: [[Val3:%.*]] = extractelement <3 x float> [[VecA3]], i32 2
// CHECK-NEXT: [[VecA3Ptr:%.*]] = getelementptr <3 x float>, ptr [[VecGep1]], i32 0, i32 2
// CHECK-NEXT: store float [[Val3]], ptr [[VecA3Ptr]], align 4

// CHECK-NEXT: [[VecB1:%.*]] = load <4 x float>, ptr [[VecGep4]], align 4
// CHECK-NEXT: [[Val4:%.*]] = extractelement <4 x float> [[VecB1]], i32 0
// CHECK-NEXT: [[VecB1Ptr:%.*]] = getelementptr <4 x float>, ptr [[VecGep2]], i32 0, i32 0
// CHECK-NEXT: store float [[Val4]], ptr [[VecB1Ptr]], align 4

// CHECK-NEXT: [[VecB2:%.*]] = load <4 x float>, ptr [[VecGep4]], align 4
// CHECK-NEXT: [[Val5:%.*]] = extractelement <4 x float> [[VecB2]], i32 1
// CHECK-NEXT: [[VecB2Ptr:%.*]] = getelementptr <4 x float>, ptr [[VecGep2]], i32 0, i32 1
// CHECK-NEXT: store float [[Val5]], ptr [[VecB2Ptr]], align 4

// CHECK-NEXT: [[VecB3:%.*]] = load <4 x float>, ptr [[VecGep4]], align 4
// CHECK-NEXT: [[Val6:%.*]] = extractelement <4 x float> [[VecB3]], i32 2
// CHECK-NEXT: [[VecB3Ptr:%.*]] = getelementptr <4 x float>, ptr [[VecGep2]], i32 0, i32 2
// CHECK-NEXT: store float [[Val6]], ptr [[VecB3Ptr]], align 4

// CHECK-NEXT: [[VecB4:%.*]] = load <4 x float>, ptr [[VecGep4]], align 4
// CHECK-NEXT: [[Val7:%.*]] = extractelement <4 x float> [[VecB4]], i32 3
// CHECK-NEXT: [[VecB4Ptr:%.*]] = getelementptr <4 x float>, ptr [[VecGep2]], i32 0, i32 3
// CHECK-NEXT: store float [[Val7]], ptr [[VecB4Ptr]], align 4

// CHECK-NEXT: ret void
void case2() {
  S localS = (S)cbs;
}
