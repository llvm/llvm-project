// RUN: %clang_cc1 -std=c++23 %s -ast-dump -triple x86_64-unknown-unknown -o - | FileCheck -strict-whitespace %s

struct Functor {
  static int operator()(int x, int y) {
    return x + y;
  }
  static int operator[](int x, int y) {
    return x + y;
  }
};

Functor& get_functor() {
  static Functor functor;
  return functor;
}

void call_static_operators() {
  Functor functor;

  int z1 = functor(1, 2);
  // CHECK:      CXXOperatorCallExpr {{.*}} 'int' '()'
  // CHECK-NEXT: |-ImplicitCastExpr {{.*}} <col:19, col:24> 'int (*)(int, int)' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} <col:19, col:24> 'int (int, int)' lvalue CXXMethod {{.*}} 'operator()' 'int (int, int)'
  // CHECK-NEXT: |-DeclRefExpr {{.*}} <col:12> 'Functor' lvalue Var {{.*}} 'functor' 'Functor'
  // CHECK-NEXT: |-IntegerLiteral {{.*}} <col:20> 'int' 1
  // CHECK-NEXT: `-IntegerLiteral {{.*}} <col:23> 'int' 2

  int z2 = functor[1, 2];
  // CHECK:      CXXOperatorCallExpr {{.*}} 'int' '[]'
  // CHECK-NEXT: |-ImplicitCastExpr {{.*}} <col:19, col:24> 'int (*)(int, int)' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} <col:19, col:24> 'int (int, int)' lvalue CXXMethod {{.*}} 'operator[]' 'int (int, int)'
  // CHECK-NEXT: |-DeclRefExpr {{.*}} <col:12> 'Functor' lvalue Var {{.*}} 'functor' 'Functor'
  // CHECK-NEXT: |-IntegerLiteral {{.*}} <col:20> 'int' 1
  // CHECK-NEXT: `-IntegerLiteral {{.*}} <col:23> 'int' 2

  int z3 = get_functor()(1, 2);
  // CHECK:      CXXOperatorCallExpr {{.*}} 'int' '()'
  // CHECK-NEXT: |-ImplicitCastExpr {{.*}} <col:25, col:30> 'int (*)(int, int)' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} <col:25, col:30> 'int (int, int)' lvalue CXXMethod {{.*}} 'operator()' 'int (int, int)'
  // CHECK-NEXT: |-CallExpr {{.*}} <col:12, col:24> 'Functor' lvalue
  // CHECK-NEXT: | `-ImplicitCastExpr {{.*}} <col:12> 'Functor &(*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: |   `-DeclRefExpr {{.*}} <col:12> 'Functor &()' lvalue Function {{.*}} 'get_functor' 'Functor &()'
  // CHECK-NEXT: |-IntegerLiteral {{.*}} <col:26> 'int' 1
  // CHECK-NEXT: `-IntegerLiteral {{.*}} <col:29> 'int' 2

  int z4 = get_functor()[1, 2];
  // CHECK:      CXXOperatorCallExpr {{.*}} 'int' '[]'
  // CHECK-NEXT: |-ImplicitCastExpr {{.*}} <col:25, col:30> 'int (*)(int, int)' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} <col:25, col:30> 'int (int, int)' lvalue CXXMethod {{.*}} 'operator[]' 'int (int, int)'
  // CHECK-NEXT: |-CallExpr {{.*}} <col:12, col:24> 'Functor' lvalue
  // CHECK-NEXT: | `-ImplicitCastExpr {{.*}} <col:12> 'Functor &(*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: |   `-DeclRefExpr {{.*}} <col:12> 'Functor &()' lvalue Function {{.*}} 'get_functor' 'Functor &()'
  // CHECK-NEXT: |-IntegerLiteral {{.*}} <col:26> 'int' 1
  // CHECK-NEXT: `-IntegerLiteral {{.*}} <col:29> 'int' 2
}
