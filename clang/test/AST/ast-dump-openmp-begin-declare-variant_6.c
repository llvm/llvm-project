// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(implementation={vendor(ibm)})
int also_after(void) {
  return 1;
}
int also_before(void) {
  return 2;
}
#pragma omp end declare variant

int also_after(void) {
  return 0;
}

int main(void) {
  // Should return 0.
  return also_after() + also_before();
}

// Make sure:
//  - we see the specialization in the AST
//  - we do use the original pointers for the calls as the variants are not applicable (this is not the ibm compiler).

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __int128_t '__int128'
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} '__int128'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __uint128_t 'unsigned __int128'
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} 'unsigned __int128'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __NSConstantString
// CHECK-NEXT: | `-typeDetails: RecordType {{.*}}
// CHECK: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __builtin_ms_va_list 'char *'
// CHECK-NEXT: | `-typeDetails: PointerType {{.*}} 'char *'
// CHECK-NEXT: |   `-typeDetails: BuiltinType {{.*}} 'char'
// CHECK-NEXT: |-TypedefDecl {{.*}} 
// CHECK-NEXT: | `-typeDetails: ConstantArrayType {{.*}} 
// CHECK: |   `-typeDetails: RecordType {{.*}} 
// CHECK: |-FunctionDecl {{.*}}
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:23, line:7:1>
// CHECK-NEXT: |   `-ReturnStmt {{.*}} <line:6:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral {{.*}} <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl {{.*}} 
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:22, line:20:1>
// CHECK-NEXT: |   `-ReturnStmt {{.*}} <line:19:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral {{.*}} <col:10> 'int' 0
// CHECK-NEXT: `-FunctionDecl {{.*}}
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:16, line:25:1>
// CHECK-NEXT:     `-ReturnStmt {{.*}} <line:24:3, col:37>
// CHECK-NEXT:       `-BinaryOperator {{.*}} <col:10, col:37> 'int' '+'
// CHECK-NEXT:         |-CallExpr {{.*}} <col:10, col:21> 'int'
// CHECK-NEXT:         | `-ImplicitCastExpr {{.*}} 
// CHECK-NEXT:         |   `-DeclRefExpr {{.*}}
// CHECK-NEXT:         `-CallExpr {{.*}} <col:25, col:37> 'int'
// CHECK-NEXT:           `-ImplicitCastExpr {{.*}}
// CHECK-NEXT:             `-DeclRefExpr {{.*}}
