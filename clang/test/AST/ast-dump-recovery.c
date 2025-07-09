// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -frecovery-ast -fno-recovery-ast-type -ast-dump %s | FileCheck -strict-whitespace %s

int some_func(int);

int unmatch_arg_call = some_func();

const int a = 1;

int postfix_inc = a++;

int unary_address = &(a + 1);

int ternary = a ? undef : a;

void test1() {
  static int foo = a++; // verify no crash on local static var decl.
}

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

void test3() {
  (*__builtin_classify_type)(1);

  extern void ext();
  ext(undef_var);
}

// Verify no crash.
void test4() {
  enum GH62446 {
    invalid_enum_value = "a" * 2,
    b,
  };
}

// Verify no crash
void test5_GH62711() {
  if (__builtin_va_arg(undef, int) << 1);
}

void test6_GH50244() {
  double array[16];
  sizeof array / sizeof foo(undef);
}

// No crash on DeclRefExpr that refers to ValueDecl with invalid initializers.
void test7() {
  int b[] = {""()};
  (unsigned) b; // GH50236

  b + 1; // GH50243
  return c(b); // GH48636
}
int test8_GH50320_b[] = {""()};
int test8 = test_8GH50320_b[0];

// CHECK:      TranslationUnitDecl 0x{{.+}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __int128_t '__int128'
// CHECK-NEXT: | `-typeDetails: BuiltinType 0x{{.+}} '__int128'
// CHECK-NEXT: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __uint128_t 'unsigned __int128'
// CHECK-NEXT: | `-typeDetails: BuiltinType 0x{{.+}} 'unsigned __int128'
// CHECK-NEXT: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __NSConstantString 'struct __NSConstantString_tag'
// CHECK-NEXT: | `-typeDetails: RecordType 0x{{.+}} 'struct __NSConstantString_tag'
// CHECK-NEXT: |   `-Record 0x{{.+}} '__NSConstantString_tag'
// CHECK-NEXT: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __builtin_ms_va_list 'char *'
// CHECK-NEXT: | `-typeDetails: PointerType 0x{{.+}} 'char *'
// CHECK-NEXT: |   `-typeDetails: BuiltinType 0x{{.+}} 'char'
// CHECK-NEXT: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __builtin_va_list 'struct __va_list_tag[1]'
// CHECK-NEXT: | `-typeDetails: ConstantArrayType 0x{{.+}} 'struct __va_list_tag[1]' 1
// CHECK-NEXT: |   `-typeDetails: RecordType 0x{{.+}} 'struct __va_list_tag'
// CHECK-NEXT: |     `-Record 0x{{.+}} '__va_list_tag'
// CHECK-NEXT: |-FunctionDecl 0x{{.+}} <{{.*}} col:{{.*}}> col:{{.*}} used some_func 'int (int)'
// CHECK-NEXT: | `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'int'
// CHECK-NEXT: |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} unmatch_arg_call 'int' cinit
// CHECK-NEXT: | |-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (int)' Function 0x{{.+}} 'some_func' 'int (int)'
// CHECK-NEXT: | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used a 'const int' cinit
// CHECK-NEXT: | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK-NEXT: |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} postfix_inc 'int' cinit
// CHECK-NEXT: | |-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int' lvalue Var 0x{{.+}} 'a' 'const int'
// CHECK-NEXT: | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} unary_address 'int' cinit
// CHECK-NEXT: | |-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: | | `-ParenExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int'
// CHECK-NEXT: | |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '+'
// CHECK-NEXT: | |     |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK-NEXT: | |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int' lvalue Var 0x{{.+}} 'a' 'const int'
// CHECK-NEXT: | |     `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} ternary 'int' cinit
// CHECK-NEXT: | |-ConditionalOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors
// CHECK-NEXT: | | |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int' lvalue Var 0x{{.+}} 'a' 'const int'
// CHECK-NEXT: | | |-RecoveryExpr 0x{{.+}} <col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int' lvalue Var 0x{{.+}} 'a' 'const int'
// CHECK-NEXT: | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test1 'void ()'
// CHECK-NEXT: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK-NEXT: |   `-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |     `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} foo 'int' static cinit
// CHECK-NEXT: |       |-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int' lvalue Var 0x{{.+}} 'a' 'const int'
// CHECK-NEXT: |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test2 'void ()'
// CHECK-NEXT: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK-NEXT: |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used ptr 'int *'
// CHECK-NEXT: |   |   `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK-NEXT: |   |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |   |-BinaryOperator 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'int *' contains-errors '='
// CHECK-NEXT: |   | |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} 'ptr' 'int *'
// CHECK-NEXT: |   | `-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (int)' Function 0x{{.+}} 'some_func' 'int (int)'
// CHECK-NEXT: |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used compoundOp 'int'
// CHECK-NEXT: |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |   |-CompoundAssignOperator 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'int' contains-errors '+=' ComputeLHSTy='NULL TYPE' ComputeResultTy='NULL TYPE'
// CHECK-NEXT: |   | |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'compoundOp' 'int'
// CHECK-NEXT: |   | `-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (int)' Function 0x{{.+}} 'some_func' 'int (int)'
// CHECK-NEXT: |   |-BinaryOperator 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'int' contains-errors '||'
// CHECK-NEXT: |   | |-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (int)' Function 0x{{.+}} 'some_func' 'int (int)'
// CHECK-NEXT: |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: |   |-BinaryOperator 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors ','
// CHECK-NEXT: |   | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: |   | `-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (int)' Function 0x{{.+}} 'some_func' 'int (int)'
// CHECK-NEXT: |   |-BinaryOperator 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'int' contains-errors ','
// CHECK-NEXT: |   | |-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (int)' Function 0x{{.+}} 'some_func' 'int (int)'
// CHECK-NEXT: |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used f 'float'
// CHECK-NEXT: |   |   `-typeDetails: BuiltinType 0x{{.+}} 'float'
// CHECK-NEXT: |   |-ParenExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors
// CHECK-NEXT: |   | `-ConditionalOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors
// CHECK-NEXT: |   |   |-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   | |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} 'ptr' 'int *'
// CHECK-NEXT: |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'float' lvalue Var 0x{{.+}} 'f' 'float'
// CHECK-NEXT: |   |   |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} 'ptr' 'int *'
// CHECK-NEXT: |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'float' lvalue Var 0x{{.+}} 'f' 'float'
// CHECK-NEXT: |   `-CStyleCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'float' contains-errors <Dependent>
// CHECK-NEXT: |     `-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |       `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (int)' Function 0x{{.+}} 'some_func' 'int (int)'
// CHECK-NEXT: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test3 'void ()'
// CHECK-NEXT: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK-NEXT: |   |-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors
// CHECK-NEXT: |   | |-ParenExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   | | `-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> '<builtin fn type>' Function 0x{{.+}} '__builtin_classify_type' 'int ()'
// CHECK-NEXT: |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |   | `-FunctionDecl 0x{{.+}} parent 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used ext 'void ()' extern
// CHECK-NEXT: |   `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors
// CHECK-NEXT: |     |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void ()' Function 0x{{.+}} 'ext' 'void ()'
// CHECK-NEXT: |     `-RecoveryExpr 0x{{.+}} <col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used __builtin_classify_type 'int ()' extern
// CHECK-NEXT: | |-attrDetails: BuiltinAttr 0x{{.+}} <<invalid sloc>> Implicit 209
// CHECK-NEXT: | |-attrDetails: NoThrowAttr 0x{{.+}} <col:{{.*}}> Implicit
// CHECK-NEXT: | `-attrDetails: ConstAttr 0x{{.+}} <col:{{.*}}> Implicit
// CHECK-NEXT: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test4 'void ()'
// CHECK-NEXT: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK-NEXT: |   `-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK-NEXT: |     `-EnumDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} GH62446
// CHECK-NEXT: |       |-EnumConstantDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} invalid_enum_value 'int'
// CHECK-NEXT: |       | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' contains-errors <IntegralCast>
// CHECK-NEXT: |       |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors <LValueToRValue>
// CHECK-NEXT: |       |     `-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |       |       |-StringLiteral 0x{{.+}} <col:{{.*}}> 'char[2]' lvalue "a"
// CHECK-NEXT: |       |       `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 2
// CHECK-NEXT: |       `-EnumConstantDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} b 'int'
// CHECK-NEXT: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test5_GH62711 'void ()'
// CHECK-NEXT: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK-NEXT: |   `-IfStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' contains-errors '<<'
// CHECK-NEXT: |     | |-VAArgExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' contains-errors
// CHECK-NEXT: |     | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> '<dependent type>' contains-errors <LValueToRValue>
// CHECK-NEXT: |     | |   `-RecoveryExpr 0x{{.+}} <col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |     | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: |     `-NullStmt 0x{{.+}} <col:{{.*}}>
// CHECK-NEXT: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test6_GH50244 'void ()'
// CHECK-NEXT: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK-NEXT: |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} referenced array 'double[16]'
// CHECK-NEXT: |   |   `-typeDetails: ConstantArrayType 0x{{.+}} 'double[16]' 16
// CHECK-NEXT: |   |     `-typeDetails: BuiltinType 0x{{.+}} 'double'
// CHECK-NEXT: |   `-BinaryOperator 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'unsigned long' contains-errors '/'
// CHECK-NEXT: |     |-UnaryExprOrTypeTraitExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'unsigned long' sizeof
// CHECK-NEXT: |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'double[16]' lvalue Var 0x{{.+}} 'array' 'double[16]' non_odr_use_unevaluated
// CHECK-NEXT: |     `-UnaryExprOrTypeTraitExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'unsigned long' contains-errors sizeof
// CHECK-NEXT: |       `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors
// CHECK-NEXT: |         |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int ()' Function 0x{{.+}} 'foo' 'int ()' non_odr_use_unevaluated
// CHECK-NEXT: |         `-RecoveryExpr 0x{{.+}} <col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test7 'void ()'
// CHECK-NEXT: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK-NEXT: |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used b 'int[]' cinit
// CHECK-NEXT: |   |   |-InitListExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'void' contains-errors
// CHECK-NEXT: |   |   | `-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: |   |   |   `-StringLiteral 0x{{.+}} <col:{{.*}}> 'char[1]' lvalue ""
// CHECK-NEXT: |   |   `-typeDetails: DependentSizedArrayType 0x{{.+}} 'int[]' dependent   <col:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |   |     |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |   |     `-<<<NULL>>>
// CHECK-NEXT: |   |-CStyleCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'unsigned int' contains-errors <Dependent>
// CHECK-NEXT: |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[]' contains-errors lvalue Var 0x{{.+}} 'b' 'int[]'
// CHECK-NEXT: |   |-BinaryOperator 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors '+'
// CHECK-NEXT: |   | |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[]' contains-errors lvalue Var 0x{{.+}} 'b' 'int[]'
// CHECK-NEXT: |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK-NEXT: |   `-ReturnStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |     `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors
// CHECK-NEXT: |       |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int ()' Function 0x{{.+}} 'c' 'int ()'
// CHECK-NEXT: |       `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[]' contains-errors lvalue Var 0x{{.+}} 'b' 'int[]'
// CHECK-NEXT: |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used test8_GH50320_b 'int[]' cinit
// CHECK-NEXT: | |-InitListExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'void' contains-errors
// CHECK-NEXT: | | `-RecoveryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<dependent type>' contains-errors lvalue
// CHECK-NEXT: | |   `-StringLiteral 0x{{.+}} <col:{{.*}}> 'char[1]' lvalue ""
// CHECK-NEXT: | `-typeDetails: DependentSizedArrayType 0x{{.+}} 'int[]' dependent   <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK-NEXT: |   |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT: |   `-<<<NULL>>>
// CHECK-NEXT: `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} test8 'int' cinit
// CHECK-NEXT:   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' contains-errors <LValueToRValue>
// CHECK-NEXT:   | `-ArraySubscriptExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' contains-errors lvalue
// CHECK-NEXT:   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' contains-errors <ArrayToPointerDecay>
// CHECK-NEXT:   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[]' contains-errors lvalue Var 0x{{.+}} 'test8_GH50320_b' 'int[]'
// CHECK-NEXT:   |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK-NEXT:   `-typeDetails: BuiltinType 0x{{.+}} 'int'