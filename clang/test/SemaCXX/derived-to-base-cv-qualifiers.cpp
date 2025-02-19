// RUN: %clang_cc1 -std=c++20 -fsyntax-only -ast-dump %s | FileCheck %s

// Ensure volatile is preserved during derived-to-base conversion. 
namespace PR127683 {

struct Base {
  int Val;
};

struct Derived : Base { };

volatile Derived Obj;

// CHECK:      |-FunctionDecl {{.*}} test_volatile_store 'void ()'
// CHECK-NEXT:   `-CompoundStmt {{.*}}
// CHECK-NEXT:     `-BinaryOperator {{.*}} 'volatile int' lvalue '='
// CHECK-NEXT:       |-MemberExpr {{.*}} 'volatile int' lvalue .Val
// CHECK-NEXT:       | `-ImplicitCastExpr {{.*}} 'volatile PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
void test_volatile_store() {
  Obj.Val = 0;
}

// CHECK:      `-FunctionDecl {{.*}} test_volatile_load 'void ()'
// CHECK-NEXT:   `-CompoundStmt {{.*}}
// CHECK-NEXT:     `-DeclStmt {{.*}}
// CHECK-NEXT:       `-VarDecl {{.*}} Val 'int' cinit
// CHECK-NEXT:         |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:         | `-MemberExpr {{.*}} 'volatile int' lvalue .Val
// CHECK-NEXT:         |   `-ImplicitCastExpr {{.*}} 'volatile PR127683::Base' lvalue <UncheckedDerivedToBase (Base)>
void test_volatile_load() {
  [[maybe_unused]] int Val = Obj.Val;
}

}
