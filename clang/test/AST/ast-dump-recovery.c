// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -frecovery-ast -fno-recovery-ast-type -ast-dump %s | FileCheck -strict-whitespace %s

int some_func(int);

// CHECK:     VarDecl {{.*}} unmatch_arg_call 'int' cinit
// CHECK-NEXT: `-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:   `-DeclRefExpr {{.*}} 'some_func'
int unmatch_arg_call = some_func();

const int a = 1;

// CHECK:     VarDecl {{.*}} postfix_inc
// CHECK-NEXT: `-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:   `-DeclRefExpr {{.*}} 'a'
int postfix_inc = a++;

// CHECK:     VarDecl {{.*}} unary_address
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  `-ParenExpr {{.*}}
// CHECK-NEXT:    `-BinaryOperator {{.*}} '+'
// CHECK-NEXT:      |-ImplicitCastExpr
// CHECK-NEXT:      | `-DeclRefExpr {{.*}} 'a'
// CHECK-NEXT:      `-IntegerLiteral {{.*}} 'int'
int unary_address = &(a + 1);

// CHECK:       VarDecl {{.*}} ternary 'int' cinit
// CHECK-NEXT:  `-ConditionalOperator {{.*}}
// CHECK-NEXT:    |-DeclRefExpr {{.*}} 'a'
// CHECK-NEXT:    |-RecoveryExpr {{.*}}
// CHECK-NEXT:    `-DeclRefExpr {{.*}} 'a'
int ternary = a ? undef : a;

void test1() {
  // CHECK:     `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:  `-DeclRefExpr {{.*}} 'a' 'const int'
  static int foo = a++; // verify no crash on local static var decl.
}

void test2() {
  int* ptr;
  // CHECK:     BinaryOperator {{.*}} 'int *' contains-errors '='
  // CHECK-NEXT: |-DeclRefExpr {{.*}} 'ptr' 'int *'
  // CHECK-NEXT: `-RecoveryExpr {{.*}}
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'some_func'
  ptr = some_func(); // should not crash

  int compoundOp;
  // CHECK:     CompoundAssignOperator {{.*}} 'int' contains-errors '+='
  // CHECK-NEXT: |-DeclRefExpr {{.*}} 'compoundOp'
  // CHECK-NEXT: `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'some_func'
  compoundOp += some_func();

  // CHECK:     BinaryOperator {{.*}} 'int' contains-errors '||'
  // CHECK-NEXT: |-RecoveryExpr {{.*}}
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} 'some_func'
  // CHECK-NEXT: `-IntegerLiteral {{.*}} 'int' 1
  some_func() || 1;

  // CHECK:     BinaryOperator {{.*}} '<dependent type>' contains-errors ','
  // CHECK-NEXT: |-IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT: `-RecoveryExpr {{.*}}
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'some_func'
  1, some_func();
  // CHECK:     BinaryOperator {{.*}} 'int' contains-errors ','
  // CHECK-NEXT: |-RecoveryExpr {{.*}} '<dependent type>'
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} 'some_func'
  // CHECK-NEXT: `-IntegerLiteral {{.*}} 'int' 1
  some_func(), 1;

  // conditional operator (comparison is invalid)
  float f;
  // CHECK:     ConditionalOperator {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: |-RecoveryExpr {{.*}} '<dependent type>'
  // CHECK-NEXT: | |-DeclRefExpr {{.*}} 'int *' lvalue
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} 'float' lvalue
  // CHECK-NEXT: |-DeclRefExpr {{.*}} 'int *' lvalue
  // CHECK-NEXT: `-DeclRefExpr {{.*}} 'float' lvalue
  (ptr > f ? ptr : f);

  // CHECK:     CStyleCastExpr {{.*}} 'float' contains-errors <Dependent>
  // CHECK-NEXT: `-RecoveryExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'some_func'
  (float)some_func();
}

void test3() {
  // CHECK:     CallExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: |-ParenExpr {{.*}} contains-errors lvalue
  // CHECK-NEXT: | `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT: |   `-DeclRefExpr {{.*}} '__builtin_classify_type'
  // CHECK-NEXT: `-IntegerLiteral {{.*}} 'int' 1
  (*__builtin_classify_type)(1);

  extern void ext();
  // CHECK:     CallExpr {{.*}} 'void' contains-errors
  // CHECK-NEXT: |-DeclRefExpr {{.*}} 'ext'
  // CHECK-NEXT: `-RecoveryExpr {{.*}} '<dependent type>'
  ext(undef_var);
}
