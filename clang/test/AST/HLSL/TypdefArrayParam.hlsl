// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -x hlsl -ast-dump %s | FileCheck %s

typedef uint4 uint32_t4;
typedef uint32_t4 uint32_t8[2];

// CHECK-LABEL: FunctionDecl {{.*}} used Accumulate 'uint32_t (uint32_t4[2])'
// CHECK-NEXT: ParmVarDecl {{.*}} used V 'uint32_t4[2]'
uint32_t Accumulate(uint32_t8 V) {
  uint32_t4 SumVec = V[0] + V[1];
  return SumVec.x + SumVec.y + SumVec.z + SumVec.w;
}

// CHECK-LABEL: FunctionDecl {{.*}} used InOutAccu 'void (inout uint32_t4[2])'
// CHECK-NEXT: ParmVarDecl {{.*}} used V 'uint32_t4[2]'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout
void InOutAccu(inout uint32_t8 V) {
  uint32_t4 SumVec = V[0] + V[1];
  V[0] = SumVec;
}

// CHECK-LABEL: call1
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(inout uint32_t4[2])' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (inout uint32_t4[2])' lvalue Function {{.*}} 'InOutAccu' 'void (inout uint32_t4[2])'
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'uint32_t4[2]' lvalue inout
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'uint32_t8':'uint32_t4[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'uint32_t8':'uint32_t4[2]' lvalue Var {{.*}} 'B' 'uint32_t8':'uint32_t4[2]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'uint32_t4[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint32_t4[2]' <HLSLArrayRValue>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'uint32_t8':'uint32_t4[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'uint32_t8':'uint32_t4[2]' lvalue Var {{.*}} 'B' 'uint32_t8':'uint32_t4[2]'
// CHECK-NEXT: BinaryOperator {{.*}} 'uint32_t8':'uint32_t4[2]' lvalue '='
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'uint32_t8':'uint32_t4[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'uint32_t8':'uint32_t4[2]' lvalue Var {{.*}} 'B' 'uint32_t8':'uint32_t4[2]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint32_t4[2]' <HLSLArrayRValue>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'uint32_t4[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint32_t4[2]' <HLSLArrayRValue>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'uint32_t8':'uint32_t4[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'uint32_t8':'uint32_t4[2]' lvalue Var {{.*}} 'B' 'uint32_t8':'uint32_t4[2]'
void call1() {
  uint32_t4 A = {1,2,3,4};
  uint32_t8 B = {A,A};
  InOutAccu(B);
}

// CHECK-LABEL: call2
// CHECK: VarDecl {{.*}} D 'uint32_t':'unsigned int' cinit
// CHECK-NEXT: CallExpr {{.*}} 'uint32_t':'unsigned int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint32_t (*)(uint32_t4[2])' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'uint32_t (uint32_t4[2])' lvalue Function {{.*}} 'Accumulate' 'uint32_t (uint32_t4[2])'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint4[2]' <HLSLArrayRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'uint4[2]' lvalue Var {{.*}} 'C' 'uint4[2]'
void call2() {
  uint4 A = {1,2,3,4};
  uint4 C[2] = {A,A};
  uint32_t D = Accumulate(C);
}

typedef int Foo[2];

// CHECK-LABEL: call3
// CHECK: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int[2]' lvalue ParmVar {{.*}} 'F' 'int[2]'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
int call3(Foo F) {
  return F[0];
}
