// RUN: %clang_cc1 -Wno-hlsl-implicit-binding -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-pch -o %t %s
// RUN: %clang_cc1 -Wno-hlsl-implicit-binding -triple dxil-pc-shadermodel6.3-library -x hlsl -include-pch %t -ast-dump-all %S/Inputs/empty.hlsl | FileCheck  %s

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
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK-NEXT: HLSLResourceBindingAttr {{.*}} Implicit "" "0"
// CHECK-NEXT: VarDecl 0x[[A:[0-9a-f]+]] {{.*}} imported used a 'hlsl_constant float'
// CHECK-NEXT: CXXRecordDecl {{.*}} imported implicit <undeserialized declarations> struct __cblayout_A definition
// CHECK: FieldDecl {{.*}} imported a 'float'

// CHECK: HLSLBufferDecl {{.*}} line:{{[0-9]+}}:9 imported <undeserialized declarations> tbuffer B
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit SRV
// CHECK-NEXT: HLSLResourceBindingAttr {{.*}} Implicit "" "0"
// CHECK-NEXT: VarDecl 0x[[B:[0-9a-f]+]] {{.*}} imported used b 'hlsl_constant float'
// CHECK-NEXT: CXXRecordDecl 0x{{[0-9a-f]+}} {{.*}} imported implicit <undeserialized declarations> struct __cblayout_B definition
// CHECK: FieldDecl 0x{{[0-9a-f]+}} {{.*}} imported b 'float'

// CHECK-NEXT: FunctionDecl {{.*}} line:{{[0-9]+}}:7 imported foo 'float ()'
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-NEXT: ReturnStmt {{.*}}
// CHECK-NEXT: BinaryOperator {{.*}} 'float' '+'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_constant float' lvalue Var 0x[[A]] 'a' 'hlsl_constant float'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_constant float' lvalue Var 0x[[B]] 'b' 'hlsl_constant float'
