// RUN: %clang_cc1 -ast-dump -triple x86_64-linux-gnu %s \
// RUN: | FileCheck %s --strict-whitespace --check-prefixes=CHECK
//
// Tests with serialization:
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck %s --strict-whitespace

int fun() {
    int x = 0;
    __builtin_static_analysis_assume(++x >= 1);
    return x;
}

// CHECK: |-CallExpr {{.*}} <line:11:5, col:46> 'void'
// CHECK: | |-ImplicitCastExpr {{.*}} <col:5> 'void (*)(_Bool)' <BuiltinFnToFnPtr>
// CHECK: | | `-DeclRefExpr {{.*}} <col:5> '<builtin fn type>' Function {{.*}} '__builtin_static_analysis_assume' 'void (_Bool)'
// CHECK: | `-ImplicitCastExpr {{.*}} <col:38, col:45> '_Bool' <IntegralToBoolean>
// CHECK: |   `-BinaryOperator {{.*}} <col:38, col:45> 'int' '>='
// CHECK: |     |-UnaryOperator {{.*}} <col:38, col:40> 'int' prefix '++'
// CHECK: |     | `-DeclRefExpr {{.*}} <col:40> 'int' lvalue Var {{.*}} 'x' 'int'
// CHECK: |     `-IntegerLiteral {{.*}} <col:45> 'int' 1
