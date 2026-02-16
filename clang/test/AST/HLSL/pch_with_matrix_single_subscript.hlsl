// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -emit-pch -o %t %S/Inputs/pch.hlsl
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -include-pch %t -ast-dump-all %s | FileCheck  %s

float3x2 gM;

// CHECK: FunctionDecl {{.*}} getRow 'float2 (uint)'
// CHECK-NEXT: ParmVarDecl {{.*}} col:20 used row 'uint':'unsigned int'
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-NEXT: ReturnStmt {{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 2>' <LValueToRValue>
// CHECK-NEXT: MatrixSingleSubscriptExpr {{.*}} 'vector<float, 2>' lvalue matrixcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_constant float3x2':'matrix<float hlsl_constant, 3, 2>' lvalue Var {{.*}} 'gM' 'hlsl_constant float3x2':'matrix<float hlsl_constant, 3, 2>'
// CHECK-NEXT: DeclRefExpr {{.*}} 'uint':'unsigned int' lvalue ParmVar {{.*}} 'row' 'uint':'unsigned int'
float2 getRow(uint row) {  
  return gM[row];
}
