// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -fnative-half-type %s -ast-dump | FileCheck %s

// Test matrix splats where the initializer scalar type differs from matrix element type.

// Bool to int matrix splat
// CHECK-LABEL: FunctionDecl {{.*}} fn0 'int4x4 (bool)'
// CHECK: ImplicitCastExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' <HLSLAggregateSplatCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <IntegralCast>
export int4x4 fn0(bool b) {
    return b;
}

// Float to int matrix splat
// CHECK-LABEL: FunctionDecl {{.*}} fn1 'int4x4 (float)'
// CHECK: ImplicitCastExpr {{.*}} 'int4x4':'matrix<int, 4, 4>' <HLSLAggregateSplatCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <FloatingToIntegral>
export int4x4 fn1(float f) {
    return f;
}

// Int to float matrix splat
// CHECK-LABEL: FunctionDecl {{.*}} fn2 'float4x4 (int)'
// CHECK: ImplicitCastExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' <HLSLAggregateSplatCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
export float4x4 fn2(int i) {
    return i;
}

// Bool to float matrix splat
// CHECK-LABEL: FunctionDecl {{.*}} fn3 'float4x4 (bool)'
// CHECK: ImplicitCastExpr {{.*}} 'float4x4':'matrix<float, 4, 4>' <HLSLAggregateSplatCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
export float4x4 fn3(bool b) {
    return b;
}
