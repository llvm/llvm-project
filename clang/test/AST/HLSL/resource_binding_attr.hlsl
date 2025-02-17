// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK:HLSLBufferDecl 0x[[CB:[0-9a-f]+]] {{.*}} line:8:9 cbuffer CB
// CHECK-NEXT:HLSLResourceClassAttr 0x[[CB:[0-9a-f]+]] {{.*}} Implicit CBuffer
// CHECK-NEXT:HLSLResourceAttr 0x[[CB:[0-9a-f]+]] {{.*}} Implicit CBuffer
// CHECK-NEXT:HLSLResourceBindingAttr 0x{{[0-9a-f]+}} <col:14> "b3" "space2"
// CHECK-NEXT:VarDecl 0x[[A:[0-9a-f]+]] {{.*}} col:9 used a 'hlsl_constant float'
cbuffer CB : register(b3, space2) {
  float a;
}

// CHECK:HLSLBufferDecl 0x[[TB:[0-9a-f]+]] {{.*}} line:17:9 tbuffer TB
// CHECK-NEXT:HLSLResourceClassAttr 0x[[CB:[0-9a-f]+]] {{.*}} Implicit SRV
// CHECK-NEXT:HLSLResourceAttr 0x[[CB:[0-9a-f]+]] {{.*}} Implicit TBuffer
// CHECK-NEXT:HLSLResourceBindingAttr 0x{{[0-9a-f]+}} <col:14> "t2" "space1"
// CHECK-NEXT:VarDecl 0x[[B:[0-9a-f]+]] {{.*}} col:9 used b 'hlsl_constant float'
tbuffer TB : register(t2, space1) {
  float b;
}

float foo() {
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} <col:10, col:14> 'float' '+'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-f]+}} <col:10> 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <col:10> 'hlsl_constant float' lvalue Var 0x[[A]] 'a' 'hlsl_constant float'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-f]+}} <col:14> 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <col:14> 'hlsl_constant float' lvalue Var 0x[[B]] 'b' 'hlsl_constant float'
  return a + b;
}

// CHECK: VarDecl 0x{{[0-9a-f]+}} <{{.*}}> col:17 UAV 'RWBuffer<float>':'hlsl::RWBuffer<float>' callinit
// CHECK-NEXT:-CXXConstructExpr 0x{{[0-9a-f]+}} <col:17> 'RWBuffer<float>':'hlsl::RWBuffer<float>' 'void ()'
// CHECK-NEXT:-HLSLResourceBindingAttr 0x{{[0-9a-f]+}} <col:23> "u3" "space0"
RWBuffer<float> UAV : register(u3);

// CHECK: -VarDecl 0x{{[0-9a-f]+}} <{{.*}}> col:17 UAV1 'RWBuffer<float>':'hlsl::RWBuffer<float>' callinit
// CHECK-NEXT:-CXXConstructExpr 0x{{[0-9a-f]+}} <col:17> 'RWBuffer<float>':'hlsl::RWBuffer<float>' 'void ()'
// CHECK-NEXT:-HLSLResourceBindingAttr 0x{{[0-9a-f]+}} <col:24> "u2" "space0"
// CHECK-NEXT:-VarDecl 0x{{[0-9a-f]+}} <col:1, col:38> col:38 UAV2 'RWBuffer<float>':'hlsl::RWBuffer<float>' callinit
// CHECK-NEXT:-CXXConstructExpr 0x{{[0-9a-f]+}} <col:38> 'RWBuffer<float>':'hlsl::RWBuffer<float>' 'void ()'
// CHECK-NEXT:-HLSLResourceBindingAttr 0x{{[0-9a-f]+}} <col:45> "u4" "space0"
RWBuffer<float> UAV1 : register(u2), UAV2 : register(u4);
