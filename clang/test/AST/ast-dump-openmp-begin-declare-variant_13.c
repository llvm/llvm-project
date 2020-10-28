// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

int also_before(void) {
  return 1;
}

#pragma omp begin declare variant match(user = {condition(1)})
int also_after(void) {
  return 0;
}
int also_before(void) {
  return 0;
}
#pragma omp end declare variant

int also_after(void) {
  return 2;
}

int test() {
  // Should return 0.
  return also_after() + also_before();
}

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit user={condition(1)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:13:1> 'int ({{.*}})' {{.*}}Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[user={condition(...)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit user={condition(1)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[user={condition(...)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:12:1> line:10:1 also_after[user={condition(...)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:12:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:11:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_6]] <line:13:1, line:15:1> line:13:1 also_before[user={condition(...)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_14:0x[a-z0-9]*]] <col:23, line:15:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_15:0x[a-z0-9]*]] <line:14:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_17:0x[a-z0-9]*]] prev [[ADDR_7]] <line:18:1, line:20:1> line:18:5 used also_after 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:22, line:20:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_19:0x[a-z0-9]*]] <line:19:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_20:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_21:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit user={condition(1)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:10:1> 'int ({{.*}})' {{.*}}Function [[ADDR_10]] 'also_after[user={condition(...)}]' 'int ({{.*}})'
// CHECK-NEXT: `-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:22:1, line:25:1> line:22:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_23:0x[a-z0-9]*]] <col:12, line:25:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_24:0x[a-z0-9]*]] <line:24:3, col:37>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_25:0x[a-z0-9]*]] <col:10, col:37> 'int' '+'
// CHECK-NEXT:         |-PseudoObjectExpr [[ADDR_26:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT:         | |-CallExpr [[ADDR_27:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT:         | | `-ImplicitCastExpr [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | |   `-DeclRefExpr [[ADDR_29:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// CHECK-NEXT:         | `-CallExpr [[ADDR_30:0x[a-z0-9]*]] <line:10:1, line:24:21> 'int'
// CHECK-NEXT:         |   `-ImplicitCastExpr [[ADDR_31:0x[a-z0-9]*]] <line:10:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |     `-DeclRefExpr [[ADDR_9]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_10]] 'also_after[user={condition(...)}]' 'int ({{.*}})'
// CHECK-NEXT:         `-PseudoObjectExpr [[ADDR_32:0x[a-z0-9]*]] <line:24:25, col:37> 'int'
// CHECK-NEXT:           |-CallExpr [[ADDR_33:0x[a-z0-9]*]] <col:25, col:37> 'int'
// CHECK-NEXT:           | `-ImplicitCastExpr [[ADDR_34:0x[a-z0-9]*]] <col:25> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:           |   `-DeclRefExpr [[ADDR_35:0x[a-z0-9]*]] <col:25> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CHECK-NEXT:           `-CallExpr [[ADDR_36:0x[a-z0-9]*]] <line:13:1, line:24:37> 'int'
// CHECK-NEXT:             `-ImplicitCastExpr [[ADDR_37:0x[a-z0-9]*]] <line:13:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:               `-DeclRefExpr [[ADDR_5]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_6]] 'also_before[user={condition(...)}]' 'int ({{.*}})'
