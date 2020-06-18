// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics
// FIXME: We have to improve the warnings here as nothing is impacted by the declare variant.
int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(device={kind(cpu)})
int also_before(void);
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(100):llvm)})
int also_after(void);
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(0):llvm)})
int also_before(void);
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
//  - we do see the ast nodes for the llvm vendor
//  - we pick the right callees

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_4:0x[a-z0-9]*]] prev [[ADDR_0]] <line:10:1, col:21> col:5 used also_before 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_5:0x[a-z0-9]*]] <line:13:1, col:20> col:5 used also_after 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_6:0x[a-z0-9]*]] prev [[ADDR_4]] <line:16:1, col:21> col:5 used also_before 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] prev [[ADDR_5]] <line:19:1, line:21:1> line:19:5 used also_after 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_8:0x[a-z0-9]*]] <col:22, line:21:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_9:0x[a-z0-9]*]] <line:20:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_10:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: `-FunctionDecl [[ADDR_11:0x[a-z0-9]*]] <line:23:1, line:26:1> line:23:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_12:0x[a-z0-9]*]] <col:12, line:26:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_13:0x[a-z0-9]*]] <line:25:3, col:37>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_14:0x[a-z0-9]*]] <col:10, col:37> 'int' '+'
// CHECK-NEXT:         |-CallExpr [[ADDR_15:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT:         | `-ImplicitCastExpr [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |   `-DeclRefExpr [[ADDR_17:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_7]] 'also_after' 'int ({{.*}})'
// CHECK-NEXT:         `-CallExpr [[ADDR_18:0x[a-z0-9]*]] <col:25, col:37> 'int'
// CHECK-NEXT:           `-ImplicitCastExpr [[ADDR_19:0x[a-z0-9]*]] <col:25> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:             `-DeclRefExpr [[ADDR_20:0x[a-z0-9]*]] <col:25> 'int ({{.*}})' {{.*}}Function [[ADDR_6]] 'also_before' 'int ({{.*}})'
