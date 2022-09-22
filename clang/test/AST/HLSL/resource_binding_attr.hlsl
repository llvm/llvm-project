// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK:HLSLBufferDecl 0x[[CB:[0-9a-f]+]] {{.*}} line:6:9 cbuffer CB
// CHECK-NEXT:HLSLResourceBindingAttr 0x{{[0-9a-f]+}} <col:14> "b3" "space2"
// CHECK-NEXT:VarDecl 0x[[A:[0-9a-f]+]] {{.*}} col:9 used a 'float'
cbuffer CB : register(b3, space2) {
  float a;
}

// CHECK:HLSLBufferDecl 0x[[TB:[0-9a-f]+]] {{.*}} line:13:9 tbuffer TB
// CHECK-NEXT:HLSLResourceBindingAttr 0x{{[0-9a-f]+}} <col:14> "t2" "space1"
// CHECK-NEXT:VarDecl 0x[[B:[0-9a-f]+]] {{.*}} col:9 used b 'float'
tbuffer TB : register(t2, space1) {
  float b;
}

float foo() {
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} <col:10, col:14> 'float' '+'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-f]+}} <col:10> 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <col:10> 'float' lvalue Var 0x[[A]] 'a' 'float'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-f]+}} <col:14> 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <col:14> 'float' lvalue Var 0x[[B]] 'b' 'float'
  return a + b;
}
