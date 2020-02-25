// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

#pragma omp begin declare variant match(device={kind(cpu)})
int also_before(void) {
  return 0;
}
#pragma omp end declare variant

int also_after(void) {
  return 0;
}

int test() {
  // Should return 0.
  return also_after() + also_before();
}

// Make sure:
//  - we do see the ast nodes for the cpu kind
//  - we pick the right callees

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, col:21> col:5 implicit used also_before 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_1:0x[a-z0-9]*]] <<invalid sloc>> Implicit device={kind(cpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_2:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_3:0x[a-z0-9]*]] 'also_before[device={kind(cpu)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_3]] <col:1, line:8:1> line:6:1 also_before[device={kind(cpu)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_4:0x[a-z0-9]*]] <col:23, line:8:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_5:0x[a-z0-9]*]] <line:7:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_6:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:11:1, line:13:1> line:11:5 used also_after 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_8:0x[a-z0-9]*]] <col:22, line:13:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_9:0x[a-z0-9]*]] <line:12:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_10:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: `-FunctionDecl [[ADDR_11:0x[a-z0-9]*]] <line:15:1, line:18:1> line:15:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_12:0x[a-z0-9]*]] <col:12, line:18:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_13:0x[a-z0-9]*]] <line:17:3, col:37>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_14:0x[a-z0-9]*]] <col:10, col:37> 'int' '+'
// CHECK-NEXT:         |-CallExpr [[ADDR_15:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT:         | `-ImplicitCastExpr [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |   `-DeclRefExpr [[ADDR_17:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_7]] 'also_after' 'int ({{.*}})'
// CHECK-NEXT:         `-PseudoObjectExpr [[ADDR_18:0x[a-z0-9]*]] <col:25, col:37> 'int'
// CHECK-NEXT:           |-CallExpr [[ADDR_19:0x[a-z0-9]*]] <col:25, col:37> 'int'
// CHECK-NEXT:           | `-ImplicitCastExpr [[ADDR_20:0x[a-z0-9]*]] <col:25> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:           |   `-DeclRefExpr [[ADDR_21:0x[a-z0-9]*]] <col:25> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CHECK-NEXT:           `-CallExpr [[ADDR_22:0x[a-z0-9]*]] <line:6:1, line:17:37> 'int'
// CHECK-NEXT:             `-ImplicitCastExpr [[ADDR_23:0x[a-z0-9]*]] <line:6:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:               `-DeclRefExpr [[ADDR_2]] <col:1> 'int ({{.*}})' Function [[ADDR_3]] 'also_before[device={kind(cpu)}]' 'int ({{.*}})'
