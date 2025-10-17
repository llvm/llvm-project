// RUN: %clang_cc1 -finclude-default-header -triple  dxil-pc-shadermodel6.6-library %s -fnative-half-type -ast-dump -ast-dump-filter=case | FileCheck %s

// CHECK-LABEL: FunctionDecl {{.*}} used case1 'uint32_t2 (uint32_t2)'
// CHECK-NEXT: ParmVarDecl {{.*}} used b 'uint32_t2':'vector<uint32_t, 2>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<uint32_t, 2>' <IntegralCast>
// CHECK-NEXT: UnaryOperator {{.*}} 'vector<bool, 2>' prefix '!' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<bool, 2>' <IntegralToBoolean>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint32_t2':'vector<uint32_t, 2>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'uint32_t2':'vector<uint32_t, 2>' lvalue ParmVar {{.*}} 'b' 'uint32_t2':'vector<uint32_t, 2>'
export uint32_t2 case1(uint32_t2 b) {
    return !b;
}

// CHECK-LABEL: FunctionDecl {{.*}} used case2 'int32_t3 (int32_t3)'
// CHECK-NEXT: ParmVarDecl {{.*}} used b 'int32_t3':'vector<int32_t, 3>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int32_t, 3>' <IntegralCast>
// CHECK-NEXT: UnaryOperator {{.*}} 'vector<bool, 3>' prefix '!' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<bool, 3>' <IntegralToBoolean>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int32_t3':'vector<int32_t, 3>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int32_t3':'vector<int32_t, 3>' lvalue ParmVar {{.*}} 'b' 'int32_t3':'vector<int32_t, 3>'
export int32_t3 case2(int32_t3 b) {
    return !b;
}

// CHECK-LABEL: FunctionDecl {{.*}} used case3 'float16_t (float16_t)'
// CHECK-NEXT: ParmVarDecl {{.*}} used b 'float16_t':'half'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float16_t':'half' <IntegralToFloating>
// CHECK-NEXT: UnaryOperator {{.*}} 'bool' prefix '!' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'bool' <FloatingToBoolean>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float16_t':'half' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float16_t':'half' lvalue ParmVar {{.*}} 'b' 'float16_t':'half'
export float16_t case3(float16_t b) {
    return !b;
}

// CHECK-LABEL: FunctionDecl {{.*}} used case4 'float4 (float4)'
// CHECK-NEXT: ParmVarDecl {{.*}} used b 'float4':'vector<float, 4>'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <IntegralToFloating>
// CHECK-NEXT: UnaryOperator {{.*}} 'vector<bool, 4>' prefix '!' cannot overflow
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<bool, 4>' <FloatingToBoolean>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4':'vector<float, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float4':'vector<float, 4>' lvalue ParmVar {{.*}} 'b' 'float4':'vector<float, 4>'
export float4 case4(float4 b) {
    return !b;
}
