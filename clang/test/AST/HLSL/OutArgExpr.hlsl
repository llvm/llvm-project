// RUN: rm -f %t.pch
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-pch -finclude-default-header -o %t.pch %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -finclude-default-header -include-pch %t.pch %s -ast-dump | FileCheck --check-prefix=AST %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -finclude-default-header -include-pch %t.pch %s -ast-print | FileCheck %s


#ifndef TEST_HLSL
#define TEST_HLSL

RWBuffer<float> Buf;

// CHECK: void trunc_Param(inout int &__restrict X) {

// AST: FunctionDecl {{.*}} used trunc_Param 'void (inout int)'
// AST-NEXT: ParmVarDecl {{.*}} X 'int &__restrict'
// AST-NEXT: HLSLParamModifierAttr {{.*}} inout

void trunc_Param(inout int X) {}

// CHECK: void zero(out int &__restrict Z) {
// CHECK-NEXT: Z = 0;

// AST: FunctionDecl {{.*}} zero 'void (out int)'
// AST-NEXT: ParmVarDecl {{.*}} used Z 'int &__restrict'
// AST-NEXT: HLSLParamModifierAttr {{.*}} out
void zero(out int Z) { Z = 0; }

// AST-LABEL: FunctionDecl {{.*}} imported used fn 'void (uint)'
// AST: CallExpr {{.*}} 'void'
// AST-NEXT: ImplicitCastExpr {{.*}} 'void (*)(inout int)' <FunctionToPointerDecay>
// AST-NEXT: DeclRefExpr {{.*}} 'void (inout int)' lvalue Function
// AST-NEXT: HLSLOutArgExpr {{.*}} 'int' lvalue inout
// AST-NEXT: OpaqueValueExpr [[LVOpV:0x[0-9a-fA-F]+]] {{.*}} 'float' lvalue
// AST-NEXT: CXXOperatorCallExpr {{.*}} 'float' lvalue '[]'
// AST-NEXT: ImplicitCastExpr {{.*}} 'float &(*)(unsigned int)' <FunctionToPointerDecay>
// AST-NEXT: DeclRefExpr {{.*}} 'float &(unsigned int)' lvalue CXXMethod {{.*}} 'operator[]' 'float &(unsigned int)'
// AST-NEXT: DeclRefExpr {{.*}} 'RWBuffer<float>':'hlsl::RWBuffer<float>' lvalue Var {{.*}} 'Buf' 'RWBuffer<float>':'hlsl::RWBuffer<float>'
// AST-NEXT: ImplicitCastExpr {{.*}} 'uint':'unsigned int' <LValueToRValue>
// AST-NEXT: DeclRefExpr {{.*}} 'uint':'unsigned int' lvalue ParmVar {{.*}} 'GI' 'uint':'unsigned int'

// AST-NEXT: OpaqueValueExpr [[TmpOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// AST-NEXT: ImplicitCastExpr {{.*}} 'int' <FloatingToIntegral>
// AST-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// AST-NEXT: OpaqueValueExpr [[LVOpV]] <col:15, col:21> 'float' lvalue

// AST: BinaryOperator {{.*}} 'float' lvalue '='
// AST-NEXT: OpaqueValueExpr [[LVOpV]] {{.*}} 'float' lvalue
// AST: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// AST-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// AST-NEXT: OpaqueValueExpr [[TmpOpV]] {{.*}} 'int' lvalue

// CHECK: void fn(uint GI) {
// CHECK:     trunc_Param(Buf[GI]);
void fn(uint GI) {
  trunc_Param(Buf[GI]);
}

#else

// AST-LABEL: FunctionDecl {{.*}} main 'void (uint)'
// AST: CallExpr {{.*}} 'void'
// AST-NEXT: ImplicitCastExpr {{.*}} 'void (*)(out int)' <FunctionToPointerDecay>
// AST-NEXT: DeclRefExpr {{.*}} 'void (out int)' lvalue Function {{.*}} 'zero' 'void (out int)'
// AST-NEXT: HLSLOutArgExpr {{.*}} 'int' lvalue out

// AST: OpaqueValueExpr [[LVOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// AST-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'I' 'int'

// AST-NEXT: OpaqueValueExpr [[TmpOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// AST-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// AST-NEXT: OpaqueValueExpr [[LVOpV]] <col:8> 'int' lvalue

// AST: BinaryOperator {{.*}} 'int' lvalue '='
// AST-NEXT: OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// AST: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// AST-NEXT: OpaqueValueExpr [[TmpOpV]] {{.*}} 'int' lvalue


[numthreads(8,1,1)]
void main(uint GI : SV_GroupIndex) {
  int I;
  zero(I);
  fn(GI);
}
#endif // TEST_HLSL
