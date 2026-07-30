// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(device={kind(gpu)})
int also_after(void) {
  return 1;
}
int also_before(void) {
  return 2;
}
#pragma omp end declare variant


#pragma omp begin declare variant match(device={kind(fpga)})

This text is never parsed!

#pragma omp end declare variant

int also_after(void) {
  return 0;
}

int test(void) {
  // Should return 0.
  return also_after() + also_before();
}

// Make sure:
//  - we do not see the ast nodes for the gpu kind
//  - we do not choke on the text in the kind(fpga) guarded scopes

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_4:0x[a-z0-9]*]] <line:25:1, line:27:1> line:25:5 used also_after 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_5:0x[a-z0-9]*]] <col:22, line:27:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_6:0x[a-z0-9]*]] <line:26:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_7:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: `-FunctionDecl [[ADDR_8:0x[a-z0-9]*]] <line:29:1, line:32:1> line:29:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_9:0x[a-z0-9]*]] <col:16, line:32:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_10:0x[a-z0-9]*]] <line:31:3, col:37>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_11:0x[a-z0-9]*]] <col:10, col:37> 'int' '+'
// CHECK-NEXT:         |-CallExpr [[ADDR_12:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT:         | `-ImplicitCastExpr [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |   `-DeclRefExpr [[ADDR_14:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_4]] 'also_after' 'int ({{.*}})'
// CHECK-NEXT:         `-CallExpr [[ADDR_15:0x[a-z0-9]*]] <col:25, col:37> 'int'
// CHECK-NEXT:           `-ImplicitCastExpr [[ADDR_16:0x[a-z0-9]*]] <col:25> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:             `-DeclRefExpr [[ADDR_17:0x[a-z0-9]*]] <col:25> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
