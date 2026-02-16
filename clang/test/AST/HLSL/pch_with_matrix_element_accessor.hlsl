// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -emit-pch -o %t %S/Inputs/pch.hlsl
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -include-pch %t -ast-dump-all %s | FileCheck  %s

float4x4 gM;

// CHECK: FunctionDecl {{.*}} getDiag 'float4 ()'
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-NEXT: ReturnStmt {{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: MatrixElementExpr {{.*}} 'vector<float hlsl_constant, 4>' lvalue _11_22_33_44
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_constant float4x4':'matrix<float hlsl_constant, 4, 4>' lvalue Var {{.*}}  'gM' 'hlsl_constant float4x4':'matrix<float hlsl_constant, 4, 4>'
float4 getDiag() {  
  return gM._11_22_33_44;
}

// CHECK: FunctionDecl {{.*}} setRowZero 'void (float4)'
// CHECK-NEXT: ParmVarDecl {{.*}} used V 'float4':'vector<float, 4>'
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-NEXT: BinaryOperator {{.*}} 'vector<float hlsl_constant, 4>' lvalue '='
// CHECK-NEXT: MatrixElementExpr {{.*}} 'vector<float hlsl_constant, 4>' lvalue _m00_m01_m02_m03
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_constant float4x4':'matrix<float hlsl_constant, 4, 4>' lvalue Var {{.*}} 'gM' 'hlsl_constant float4x4':'matrix<float hlsl_constant, 4, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'V' 'float4':'vector<float, 4>'
void setRowZero(float4 V) {  
  gM._m00_m01_m02_m03 = V;
}
