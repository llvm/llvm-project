// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c++ -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck -strict-whitespace %s

void test_type_trait() {
  // An unary type trait.
  enum E {};
  (void) __is_enum(E);
  // A binary type trait.
  (void) __is_same(int ,float);
  // An n-ary type trait.
  (void) __is_constructible(int, int, int, int);
}

void test_array_type_trait() {
  // An array type trait.
  (void) __array_rank(int[10][20]);
}

void test_expression_trait() {
  // An expression trait.
  (void) __is_lvalue_expr(1);
}

void test_unary_expr_or_type_trait() {
  // Some UETTs.
  (void) sizeof(int);
  (void) alignof(int);
  (void) __alignof(int);
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __int128_t '__int128'
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} '__int128'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __uint128_t 'unsigned __int128'
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} 'unsigned __int128'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __NSConstantString '__NSConstantString_tag'
// CHECK-NEXT: | `-typeDetails: RecordType {{.*}} '__NSConstantString_tag'
// CHECK-NEXT: |   `-CXXRecord {{.*}} '__NSConstantString_tag'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __builtin_ms_va_list 'char *'
// CHECK-NEXT: | `-typeDetails: PointerType {{.*}} 'char *'
// CHECK-NEXT: |   `-typeDetails: BuiltinType {{.*}} 'char'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __builtin_va_list '__va_list_tag[1]'
// CHECK-NEXT: | `-typeDetails: ConstantArrayType {{.*}} '__va_list_tag[1]' 1 
// CHECK-NEXT: |   `-typeDetails: RecordType {{.*}} '__va_list_tag'
// CHECK-NEXT: |     `-CXXRecord {{.*}} '__va_list_tag'
// CHECK-NEXT: |-FunctionDecl {{.*}} test_type_trait 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | `-EnumDecl {{.*}} referenced E
// CHECK-NEXT: |   |-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |   | `-TypeTraitExpr {{.*}} 'bool' __is_enum
// CHECK-NEXT: |   |   `-typeDetails: ElaboratedType {{.*}} 'E' sugar
// CHECK-NEXT: |   |     `-typeDetails: EnumType {{.*}} 'E'
// CHECK-NEXT: |   |       `-Enum {{.*}} 'E'
// CHECK-NEXT: |   |-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |   | `-TypeTraitExpr {{.*}} 'bool' __is_same
// CHECK-NEXT: |   |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   |   `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK-NEXT: |   `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |     `-TypeTraitExpr {{.*}} 'bool' __is_constructible
// CHECK-NEXT: |       |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |-FunctionDecl {{.*}} test_array_type_trait 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |     `-ArrayTypeTraitExpr {{.*}} '__size_t':'unsigned long' __array_rank
// CHECK-NEXT: |-FunctionDecl {{.*}} test_expression_trait 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |     `-ExpressionTraitExpr {{.*}} 'bool' __is_lvalue_expr
// CHECK-NEXT: `-FunctionDecl {{.*}} test_unary_expr_or_type_trait 'void ()'
// CHECK-NEXT:   `-CompoundStmt {{.*}} 
// CHECK-NEXT:     |-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT:     | `-UnaryExprOrTypeTraitExpr {{.*}} '__size_t':'unsigned long' sizeof 'int'
// CHECK-NEXT:     |-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT:     | `-UnaryExprOrTypeTraitExpr {{.*}} '__size_t':'unsigned long' alignof 'int'
// CHECK-NEXT:     `-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT:       `-UnaryExprOrTypeTraitExpr {{.*}} '__size_t':'unsigned long' __alignof 'int'

