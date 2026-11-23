// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump -std=c++20 %s | FileCheck %s

void foo(int a) {
  auto f = [](int(&&)[]) {};
  f({a});
}

// Make sure the MaterializeTemporaryExpr has a complete type, which is then
// cast to an incomplete array type.

// CHECK:         `-ExprWithCleanups 0x{{[^ ]*}}{{[^ ]*}} <line:5:3, col:8> 'void'
// CHECK-NEXT:      `-CXXOperatorCallExpr 0x{{[^ ]*}} <col:3, col:8> 'void' '()'
// CHECK-NEXT:        |-ImplicitCastExpr 0x{{[^ ]*}} <col:4, col:8> 'void (*)(int (&&)[]) const' <FunctionToPointerDecay>
// CHECK-NEXT:        | `-DeclRefExpr 0x{{[^ ]*}} <col:4, col:8> 'void (int (&&)[]) const' lvalue CXXMethod 0x{{[^ ]*}} 'operator()' 'void (int (&&)[]) const'
// CHECK-NEXT:        |-ImplicitCastExpr 0x{{[^ ]*}} <col:3> 'const (lambda at {{.*}})' lvalue <NoOp>
// CHECK-NEXT:        | `-DeclRefExpr 0x{{[^ ]*}} <col:3> '(lambda at {{.*}})' lvalue Var 0x{{[^ ]*}} 'f' '(lambda at {{.*}})'
// CHECK-NEXT:        `-ImplicitCastExpr 0x{{[^ ]*}} <col:5, col:7> 'int[]' xvalue <NoOp>
// CHECK-NEXT:          `-MaterializeTemporaryExpr 0x{{[^ ]*}} <col:5, col:7> 'int[1]' xvalue
// CHECK-NEXT:            `-InitListExpr 0x{{[^ ]*}} <col:5, col:7> 'int[1]'
// CHECK-NEXT:              `-ImplicitCastExpr 0x{{[^ ]*}} <col:6> 'int' <LValueToRValue>
// CHECK-NEXT:                `-DeclRefExpr 0x{{[^ ]*}} <col:6> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
