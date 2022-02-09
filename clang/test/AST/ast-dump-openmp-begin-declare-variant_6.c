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

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(ibm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:13:1> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[implementation={vendor(ibm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(ibm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[implementation={vendor(ibm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:12:1> line:10:1 also_after[implementation={vendor(ibm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:12:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:11:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |-FunctionDecl [[ADDR_6]] <line:13:1, line:15:1> line:13:1 also_before[implementation={vendor(ibm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_14:0x[a-z0-9]*]] <col:23, line:15:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_15:0x[a-z0-9]*]] <line:14:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: |-FunctionDecl [[ADDR_17:0x[a-z0-9]*]] prev [[ADDR_7]] <line:18:1, line:20:1> line:18:5 used also_after 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:22, line:20:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_19:0x[a-z0-9]*]] <line:19:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_20:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_21:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(ibm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:10:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(ibm)}]' 'int ({{.*}})'
// CHECK-NEXT: `-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:22:1, line:25:1> line:22:5 main 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_23:0x[a-z0-9]*]] <col:16, line:25:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_24:0x[a-z0-9]*]] <line:24:3, col:37>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_25:0x[a-z0-9]*]] <col:10, col:37> 'int' '+'
// CHECK-NEXT:         |-CallExpr [[ADDR_26:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT:         | `-ImplicitCastExpr [[ADDR_27:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |   `-DeclRefExpr [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// CHECK-NEXT:         `-CallExpr [[ADDR_29:0x[a-z0-9]*]] <col:25, col:37> 'int'
// CHECK-NEXT:           `-ImplicitCastExpr [[ADDR_30:0x[a-z0-9]*]] <col:25> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:             `-DeclRefExpr [[ADDR_31:0x[a-z0-9]*]] <col:25> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
