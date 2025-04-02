// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl \
// RUN:   -include-pch %t -ast-dump-all %S/Inputs/empty.hlsl \
// RUN: | FileCheck  %s

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
// CHECK: HLSLBufferDecl {{.*}} line:7:9 imported <undeserialized declarations> cbuffer A
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK-NEXT: HLSLResourceAttr {{.*}} Implicit CBuffer
// CHECK-NEXT: VarDecl 0x[[A:[0-9a-f]+]] {{.*}} imported used a 'hlsl_constant float'
// CHECK-NEXT: CXXRecordDecl {{.*}} imported implicit <undeserialized declarations> class __layout_A definition
// CHECK: FieldDecl {{.*}} imported a 'float'

// CHECK: HLSLBufferDecl {{.*}} line:11:9 imported <undeserialized declarations> tbuffer B
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit SRV
// CHECK-NEXT: HLSLResourceAttr {{.*}} Implicit TBuffer
// CHECK-NEXT: VarDecl 0x[[B:[0-9a-f]+]] {{.*}} imported used b 'hlsl_constant float'
// CHECK-NEXT: CXXRecordDecl 0x{{[0-9a-f]+}} {{.*}} imported implicit <undeserialized declarations> class __layout_B definition
// CHECK: FieldDecl 0x{{[0-9a-f]+}} {{.*}} imported b 'float'

// CHECK-NEXT: FunctionDecl {{.*}} line:15:7 imported foo 'float ()'
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-NEXT: ReturnStmt {{.*}}
// CHECK-NEXT: BinaryOperator {{.*}} 'float' '+'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_constant float' lvalue Var 0x[[A]] 'a' 'hlsl_constant float'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_constant float' lvalue Var 0x[[B]] 'b' 'hlsl_constant float'
