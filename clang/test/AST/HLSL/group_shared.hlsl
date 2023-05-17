// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s | FileCheck %s

//CHECK:VarDecl 0x[[A:[0-9a-f]+]] <{{.*}} col:24> col:20 used a 'groupshared float[10]'
 groupshared float a[10];

// CHECK:FunctionDecl 0x{{[0-9a-f]+}} <{{.*}}> line:[[@LINE+2]]:7 main 'void ()'
 [numthreads(8,8,1)]
 void main() {
// CHECK:BinaryOperator 0x{{[0-9a-f]+}} <{{.*}}> 'groupshared float' lvalue '='
// CHECK:ArraySubscriptExpr 0x{{[0-9a-f]+}} <col:4, col:7> 'groupshared float' lvalue
// CHECK:ImplicitCastExpr 0x{{[0-9a-f]+}} <col:4> 'groupshared float *' <ArrayToPointerDecay>
// CHECK:DeclRefExpr 0x{{[0-9a-f]+}} <col:4> 'groupshared float[10]' lvalue Var 0x[[A]] 'a' 'groupshared float[10]'
// CHECK:IntegerLiteral 0x{{[0-9a-f]+}} <col:6> 'int' 0
// CHECK:ImplicitCastExpr 0x{{[0-9a-f]+}} <col:11> 'float' <IntegralToFloating>
// CHECK:IntegerLiteral 0x{{[0-9a-f]+}} <col:11> 'int' 1
   a[0] = 1;
 }


// CHECK:HLSLNumThreadsAttr 0x{{[0-9a-f]+}} <{{.*}}> 8 8 1
