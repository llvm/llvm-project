// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -frecovery-ast -fno-recovery-ast-type -ast-dump %s | FileCheck -strict-whitespace %s

int some_func(int);
int unmatch_arg_call = some_func();
const int a = 1;
int postfix_inc = a++;
int unary_address = &(a + 1);

// CHECK: |-FunctionDecl {{.*}} used some_func 'int (int)'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} 'int'
// CHECK-NEXT: |   `-typeDetails: BuiltinType {{.*}} 'int'

// CHECK-NEXT: |-VarDecl {{.*}} unmatch_arg_call 'int' cinit
// CHECK-NEXT: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: | | `-DeclRefExpr {{.*}} 'int (int)' Function {{.*}} 'some_func' 'int (int)'
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} 'int'

// CHECK-NEXT: |-VarDecl {{.*}} used a 'const int' cinit
// CHECK-NEXT: | |-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |   `-typeDetails: BuiltinType {{.*}} 'int'

// CHECK-NEXT: |-VarDecl {{.*}} postfix_inc 'int' cinit
// CHECK-NEXT: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: | | `-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int'
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} 'int'

// CHECK-NEXT: |-VarDecl {{.*}} unary_address 'int' cinit
// CHECK-NEXT: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: | | `-ParenExpr {{.*}} 'int'
// CHECK-NEXT: | |   `-BinaryOperator {{.*}} 'int' '+'
// CHECK-NEXT: | |     |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: | |     | `-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int'
// CHECK-NEXT: | |     `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} 'int'

void test1() {
  static int foo = a++; // verify no crash on local static var decl.
}

// CHECK-NEXT: |-FunctionDecl {{.*}} test1 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-DeclStmt {{.*}} 
// CHECK-NEXT: |     `-VarDecl {{.*}} foo 'int' static cinit
// CHECK-NEXT: |       |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |       | `-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int'
// CHECK-NEXT: |       `-typeDetails: BuiltinType {{.*}} 'int'

void test2() {
  int* ptr;
  ptr = some_func(); // should not crash
  int compoundOp;
  compoundOp += some_func();
  some_func() || 1;
  1, some_func();
  some_func(), 1;

  // conditional operator (comparison is invalid)
  float f;
  (ptr > f ? ptr : f);
  (float)some_func();
}

// CHECK-NEXT: |-FunctionDecl {{.*}} test2 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | `-VarDecl {{.*}} used ptr 'int *'
// CHECK-NEXT: |   |   `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |   |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   |-BinaryOperator {{.*}} 'int *' contains-errors '='
// CHECK-NEXT: |   | |-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} 'ptr' 'int *'
// CHECK-NEXT: |   | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   `-DeclRefExpr {{.*}} 'int (int)' Function {{.*}} 'some_func' 'int (int)'
// CHECK-NEXT: |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | `-VarDecl {{.*}} used compoundOp 'int'
// CHECK-NEXT: |   |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   |-CompoundAssignOperator {{.*}} 'int' contains-errors '+=' ComputeLHSTy='NULL TYPE' ComputeResultTy='NULL TYPE'
// CHECK-NEXT: |   | |-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'compoundOp' 'int'
// CHECK-NEXT: |   | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   `-DeclRefExpr {{.*}} 'int (int)' Function {{.*}} 'some_func' 'int (int)'
// CHECK-NEXT: |   |-BinaryOperator {{.*}} 'int' contains-errors '||'
// CHECK-NEXT: |   | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   | | `-DeclRefExpr {{.*}} 'int (int)' Function {{.*}} 'some_func' 'int (int)'
// CHECK-NEXT: |   | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: |   |-BinaryOperator {{.*}} '<dependent type>' contains-errors ','
// CHECK-NEXT: |   | |-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: |   | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   `-DeclRefExpr {{.*}} 'int (int)' Function {{.*}} 'some_func' 'int (int)'
// CHECK-NEXT: |   |-BinaryOperator {{.*}} 'int' contains-errors ','
// CHECK-NEXT: |   | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   | | `-DeclRefExpr {{.*}} 'int (int)' Function {{.*}} 'some_func' 'int (int)'
// CHECK-NEXT: |   | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | `-VarDecl {{.*}} used f 'float'
// CHECK-NEXT: |   |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: |   |-ParenExpr {{.*}} '<dependent type>' contains-errors
// CHECK-NEXT: |   | `-ConditionalOperator {{.*}} '<dependent type>' contains-errors
// CHECK-NEXT: |   |   |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   | |-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} 'ptr' 'int *'
// CHECK-NEXT: |   |   | `-DeclRefExpr {{.*}} 'float' lvalue Var {{.*}} 'f' 'float'
// CHECK-NEXT: |   |   |-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} 'ptr' 'int *'
// CHECK-NEXT: |   |   `-DeclRefExpr {{.*}} 'float' lvalue Var {{.*}} 'f' 'float'
// CHECK-NEXT: |   `-CStyleCastExpr {{.*}} 'float' contains-errors <Dependent>
// CHECK-NEXT: |     `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'int (int)' Function {{.*}} 'some_func' 'int (int)'

void test3() {
  (*__builtin_classify_type)(1);
}

// CHECK-NEXT: |-FunctionDecl {{.*}} test3 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK-NEXT: |     |-ParenExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |     | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} '<builtin fn type>' Function {{.*}} '__builtin_classify_type' 'int ()'
// CHECK-NEXT: |     `-IntegerLiteral {{.*}} 'int' 1

// CHECK-NEXT: |-FunctionDecl {{.*}} implicit used __builtin_classify_type 'int ()' extern
// CHECK-NEXT: | |-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 212
// CHECK-NEXT: | |-attrDetails: NoThrowAttr {{.*}} Implicit
// CHECK-NEXT: | `-attrDetails: ConstAttr {{.*}} Implicit

// Verify no crash.
void test4() {
  enum GH62446 {
    invalid_enum_value = "a" * 2,
    b,
  };
}

// CHECK-NEXT: |-FunctionDecl {{.*}} test4 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-DeclStmt {{.*}} 
// CHECK-NEXT: |     `-EnumDecl {{.*}} GH62446
// CHECK-NEXT: |       |-EnumConstantDecl {{.*}} invalid_enum_value 'int'
// CHECK-NEXT: |       | `-ImplicitCastExpr {{.*}} 'int' contains-errors <IntegralCast>
// CHECK-NEXT: |       |   `-ImplicitCastExpr {{.*}} '<dependent type>' contains-errors <LValueToRValue>
// CHECK-NEXT: |       |     `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |       |       |-StringLiteral {{.*}} 'char[2]' lvalue "a"
// CHECK-NEXT: |       |       `-IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: |       `-EnumConstantDecl {{.*}} b 'int'

// No crash on DeclRefExpr that refers to ValueDecl with invalid initializers.
void test7() {
  int b[] = {""()};

  (unsigned) b; // GH50236
  b + 1; // GH50243
  return c(b); // GH48636
}

// CHECK-NEXT: |-FunctionDecl {{.*}} test7 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | `-VarDecl {{.*}} used b 'int[]' cinit
// CHECK-NEXT: |   |   |-InitListExpr {{.*}} 'void' contains-errors
// CHECK-NEXT: |   |   | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   |   `-StringLiteral {{.*}} 'char[1]' lvalue ""
// CHECK-NEXT: |   |   `-typeDetails: DependentSizedArrayType {{.*}} 'int[]' dependent 
// CHECK-NEXT: |   |     |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   |     `-<<<NULL>>>
// CHECK-NEXT: |   |-CStyleCastExpr {{.*}} 'unsigned int' contains-errors <Dependent>
// CHECK-NEXT: |   | `-DeclRefExpr {{.*}} 'int[]' contains-errors lvalue Var {{.*}} 'b' 'int[]'
// CHECK-NEXT: |   |-BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
// CHECK-NEXT: |   | |-DeclRefExpr {{.*}} 'int[]' contains-errors lvalue Var {{.*}} 'b' 'int[]'
// CHECK-NEXT: |   | `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: |   `-ReturnStmt {{.*}} 
// CHECK-NEXT: |     `-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK-NEXT: |       |-DeclRefExpr {{.*}} 'int ()' Function {{.*}} 'c' 'int ()'
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'int[]' contains-errors lvalue Var {{.*}} 'b' 'int[]'

int test8_GH50320_b[] = {""()};
int test8 = test_8GH50320_b[0];

// CHECK: |-VarDecl {{.*}} used test8_GH50320_b 'int[]' cinit
// CHECK-NEXT: | |-InitListExpr {{.*}} 'void' contains-errors
// CHECK-NEXT: | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT: | |   `-StringLiteral {{.*}} 'char[1]' lvalue ""
// CHECK-NEXT: | `-typeDetails: DependentSizedArrayType {{.*}} 'int[]' dependent 
// CHECK-NEXT: |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   `-<<<NULL>>>
// CHECK-NEXT: `-VarDecl {{.*}} test8 'int' cinit
// CHECK-NEXT:   |-ImplicitCastExpr {{.*}} 'int' contains-errors <LValueToRValue>
// CHECK-NEXT:   | `-ArraySubscriptExpr {{.*}} 'int' contains-errors lvalue
// CHECK-NEXT:   |   |-ImplicitCastExpr {{.*}} 'int *' contains-errors <ArrayToPointerDecay>
// CHECK-NEXT:   |   | `-DeclRefExpr {{.*}} 'int[]' contains-errors lvalue Var {{.*}} 'test8_GH50320_b' 'int[]'
// CHECK-NEXT:   |   `-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:   `-typeDetails: BuiltinType {{.*}} 'int'