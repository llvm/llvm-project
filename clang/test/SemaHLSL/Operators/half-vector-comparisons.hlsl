// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -ast-dump -ast-dump-filter=test | FileCheck %s

// Regression test for issue llvm/llvm-project#213814

// CHECK-LABEL: FunctionDecl {{.*}} test_lt 'int4 (half4, half4)'
// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' '<'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'half4':'vector<half, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'b' 'half4':'vector<half, 4>'
int4 test_lt(half4 a, half4 b) {
    return a < b;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_le 'int4 (half4, half4)'
// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' '<='
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'half4':'vector<half, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'b' 'half4':'vector<half, 4>'
int4 test_le(half4 a, half4 b) {
    return a <= b;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_gt 'int4 (half4, half4)'
// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' '>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'half4':'vector<half, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'b' 'half4':'vector<half, 4>'
int4 test_gt(half4 a, half4 b) {
    return a > b;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_ge 'int4 (half4, half4)'
// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' '>='
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'half4':'vector<half, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'b' 'half4':'vector<half, 4>'
int4 test_ge(half4 a, half4 b) {
    return a >= b;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_eq 'int4 (half4, half4)'
// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' '=='
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'half4':'vector<half, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'b' 'half4':'vector<half, 4>'
int4 test_eq(half4 a, half4 b) {
    return a == b;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_ne 'int4 (half4, half4)'
// CHECK: BinaryOperator {{.*}} 'vector<int, 4>' '!='
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'a' 'half4':'vector<half, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half4':'vector<half, 4>' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'b' 'half4':'vector<half, 4>'
int4 test_ne(half4 a, half4 b) {
    return a != b;
}
