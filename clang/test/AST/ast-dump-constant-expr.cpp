// Test without serialization:
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-unknown -include-pch %t \
// RUN: -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s

// FIXME: ASTRecordReader::readAPValue and ASTRecordWriter::AddAPValue
// just give up on some APValue kinds! This *really* should be fixed.

struct array_holder {
  int i[2];
};

struct S {
  int i = 42;
};

union U {
  int i = 42;
  float f;
};

struct SU {
  S s[2];
  U u[3];
};

consteval int test_Int() { return 42; }
consteval float test_Float() { return 1.0f; }
consteval _Complex int test_ComplexInt() { return 1+2i; }
consteval _Complex float test_ComplexFloat() { return 1.2f+3.4fi; }
consteval __int128 test_Int128() { return (__int128)0xFFFFFFFFFFFFFFFF + (__int128)1; }
// FIXME: consteval array_holder test_Array() { return array_holder(); }
// FIXME: consteval S test_Struct() { return S(); }
// FIXME: consteval U test_Union() { return U(); }
// FIXME: consteval SU test_SU() { return SU(); }

void Test() {
  (void) test_Int();
  (void) test_Float();
  (void) test_ComplexInt();
  (void) test_ComplexFloat();
  (void) test_Int128();
  //(void) test_Array();
  //(void) test_Struct();
  //(void) test_Union();
  //(void) test_SU();
}
// CHECK:Dumping Test:
// CHECK-NEXT:FunctionDecl {{.*}} <{{.*}}ast-dump-constant-expr.cpp:42:1, line:52:1> line:42:6{{( imported)?}} Test 'void ()'
// CHECK-NEXT:`-CompoundStmt {{.*}} <col:13, line:52:1>
// CHECK-NEXT:  |-CStyleCastExpr {{.*}} <line:43:3, col:19> 'void' <ToVoid>
// CHECK-NEXT:  | `-ConstantExpr {{.*}} <col:10, col:19> 'int'
// CHECK-NEXT:  |   |-value: Int 42
// CHECK-NEXT:  |   `-CallExpr {{.*}} <col:10, col:19> 'int'
// CHECK-NEXT:  |     `-ImplicitCastExpr {{.*}} <col:10> 'int (*)()' <FunctionToPointerDecay>
// CHECK-NEXT:  |       `-DeclRefExpr {{.*}} <col:10> 'int ()' lvalue Function {{.*}} 'test_Int' 'int ()'
// CHECK-NEXT:  |-CStyleCastExpr {{.*}} <line:44:3, col:21> 'void' <ToVoid>
// CHECK-NEXT:  | `-ConstantExpr {{.*}} <col:10, col:21> 'float'
// CHECK-NEXT:  |   |-value: Float 1.000000e+00
// CHECK-NEXT:  |   `-CallExpr {{.*}} <col:10, col:21> 'float'
// CHECK-NEXT:  |     `-ImplicitCastExpr {{.*}} <col:10> 'float (*)()' <FunctionToPointerDecay>
// CHECK-NEXT:  |       `-DeclRefExpr {{.*}} <col:10> 'float ()' lvalue Function {{.*}} 'test_Float' 'float ()'
// CHECK-NEXT:  |-CStyleCastExpr {{.*}} <line:45:3, col:26> 'void' <ToVoid>
// CHECK-NEXT:  | `-ConstantExpr {{.*}} <col:10, col:26> '_Complex int'
// CHECK-NEXT:  |   |-value: ComplexInt 1 + 2i
// CHECK-NEXT:  |   `-CallExpr {{.*}} <col:10, col:26> '_Complex int'
// CHECK-NEXT:  |     `-ImplicitCastExpr {{.*}} <col:10> '_Complex int (*)()' <FunctionToPointerDecay>
// CHECK-NEXT:  |       `-DeclRefExpr {{.*}} <col:10> '_Complex int ()' lvalue Function {{.*}} 'test_ComplexInt' '_Complex int ()'
// CHECK-NEXT:  |-CStyleCastExpr {{.*}} <line:46:3, col:28> 'void' <ToVoid>
// CHECK-NEXT:  | `-ConstantExpr {{.*}} <col:10, col:28> '_Complex float'
// CHECK-NEXT:  |   |-value: ComplexFloat 1.200000e+00 + 3.400000e+00i
// CHECK-NEXT:  |   `-CallExpr {{.*}} <col:10, col:28> '_Complex float'
// CHECK-NEXT:  |     `-ImplicitCastExpr {{.*}} <col:10> '_Complex float (*)()' <FunctionToPointerDecay>
// CHECK-NEXT:  |       `-DeclRefExpr {{.*}} <col:10> '_Complex float ()' lvalue Function {{.*}} 'test_ComplexFloat' '_Complex float ()'
// CHECK-NEXT:  `-CStyleCastExpr {{.*}} <line:47:3, col:22> 'void' <ToVoid>
// CHECK-NEXT:    `-ConstantExpr {{.*}} <col:10, col:22> '__int128'
// CHECK-NEXT:      |-value: Int 18446744073709551616
// CHECK-NEXT:      `-CallExpr {{.*}} <col:10, col:22> '__int128'
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} <col:10> '__int128 (*)()' <FunctionToPointerDecay>
// CHECK-NEXT:          `-DeclRefExpr {{.*}} <col:10> '__int128 ()' lvalue Function {{.*}} 'test_Int128' '__int128 ()'
