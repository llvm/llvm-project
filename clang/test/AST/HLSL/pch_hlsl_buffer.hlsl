// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-pch -o %t %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -include-pch %t -ast-dump-all %S/Inputs/empty.hlsl | FileCheck  %s

cbuffer A {
  float a;
}

tbuffer B {
  float b;
}

float foo() {
  return a + b;
}

// Make sure cbuffer/tbuffer works for PCH.
// CHECK: HLSLBufferDecl {{.*}} line:{{[0-9]+}}:9 imported <undeserialized declarations> cbuffer A
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"
// CHECK: VarDecl 0x[[A:[0-9a-f]+]] {{.*}} imported used a 'hlsl_constant float'
// CHECK: CXXRecordDecl {{.*}} imported implicit <undeserialized declarations> struct __cblayout_A definition
// CHECK: FieldDecl {{.*}} imported a 'float'

// CHECK: HLSLBufferDecl {{.*}} line:{{[0-9]+}}:9 imported <undeserialized declarations> tbuffer B
// CHECK: HLSLResourceClassAttr {{.*}} Implicit SRV
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"
// CHECK: VarDecl 0x[[B:[0-9a-f]+]] {{.*}} imported used b 'hlsl_constant float'
// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} {{.*}} imported implicit <undeserialized declarations> struct __cblayout_B definition
// CHECK: FieldDecl 0x{{[0-9a-f]+}} {{.*}} imported b 'float'

// CHECK: FunctionDecl {{.*}} line:{{[0-9]+}}:7 imported foo 'float ()'
// CHECK: CompoundStmt {{.*}}
// CHECK: ReturnStmt {{.*}}
// CHECK: BinaryOperator {{.*}} 'float' '+'
// CHECK: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK: DeclRefExpr {{.*}} 'hlsl_constant float' lvalue Var 0x[[A]] 'a' 'hlsl_constant float'
// CHECK: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK: DeclRefExpr {{.*}} 'hlsl_constant float' lvalue Var 0x[[B]] 'b' 'hlsl_constant float'