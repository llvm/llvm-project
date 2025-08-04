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
}

// Verify no crash.
void test4() {
  enum GH62446 {
    // CHECK:      RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
    // CHECK-NEXT: |-StringLiteral {{.*}} "a"
    // CHECK-NEXT: `-IntegerLiteral {{.*}} 2
    invalid_enum_value = "a" * 2,
    b,
  };
}

// No crash on DeclRefExpr that refers to ValueDecl with invalid initializers.
void test7() {
  int b[] = {""()};

  // CHECK:      CStyleCastExpr {{.*}} 'unsigned int' contains-errors
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} 'int[]' contains-errors
  (unsigned) b; // GH50236

  // CHECK:      BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
  // CHECK-NEXT: |-DeclRefExpr {{.*}} 'int[]' contains-errors
  // CHECK-NEXT: `-IntegerLiteral {{.*}}
  b + 1; // GH50243

  // CHECK:      CallExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: |-DeclRefExpr {{.*}} 'int ()' Function
  // CHECK-NEXT: `-DeclRefExpr {{.*}} 'int[]' contains-errors
  return c(b); // GH48636
}
int test8_GH50320_b[] = {""()};
// CHECK: ArraySubscriptExpr {{.*}} 'int' contains-errors lvalue
int test8 = test_8GH50320_b[0];
