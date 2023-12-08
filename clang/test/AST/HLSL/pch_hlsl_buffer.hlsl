// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl \
// RUN:   -include-pch %t -fsyntax-only -ast-dump-all %S/Inputs/empty.hlsl \
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
// CHECK:HLSLBufferDecl 0x{{[0-9a-f]+}} <{{.*}}:7:1, line:9:1> line:7:9 imported <undeserialized declarations> cbuffer A
// CHECK-NEXT:`-VarDecl 0x[[A:[0-9a-f]+]] <line:8:3, col:9> col:9 imported used a 'float'
// CHECK-NEXT:HLSLBufferDecl 0x{{[0-9a-f]+}} <line:11:1, line:13:1> line:11:9 imported <undeserialized declarations> tbuffer B
// CHECK-NEXT:`-VarDecl 0x[[B:[0-9a-f]+]] <line:12:3, col:9> col:9 imported used b 'float'
// CHECK-NEXT:FunctionDecl 0x{{[0-9a-f]+}} <line:15:1, line:17:1> line:15:7 imported foo 'float ()'
// CHECK-NEXT:CompoundStmt 0x{{[0-9a-f]+}} <col:13, line:17:1>
// CHECK-NEXT:ReturnStmt 0x{{[0-9a-f]+}} <line:16:3, col:14>
// CHECK-NEXT:BinaryOperator 0x{{[0-9a-f]+}} <col:10, col:14> 'float' '+'
// CHECK-NEXT:ImplicitCastExpr 0x{{[0-9a-f]+}} <col:10> 'float' <LValueToRValue>
// CHECK-NEXT:`-DeclRefExpr 0x{{[0-9a-f]+}} <col:10> 'float' lvalue Var 0x[[A]] 'a' 'float'
// CHECK-NEXT:`-ImplicitCastExpr 0x{{[0-9a-f]+}} <col:14> 'float' <LValueToRValue>
// CHECK-NEXT:`-DeclRefExpr 0x{{[0-9a-f]+}} <col:14> 'float' lvalue Var 0x[[B]] 'b' 'float'
