// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

#pragma omp begin declare variant match(device={kind(cpu)})
int also_before(void) {
  return 0;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(implementation={vendor(score(100):llvm)})
int also_after(void) {
  return 0;
}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(0):llvm)})
int also_before(void) {
  return 1;
}
#pragma omp end declare variant

int also_after(void) {
  return 2;
}

int test() {
  // Should return 0.
  return also_after() + also_before();
}

// Make sure:
//  - we do see the ast nodes for the cpu kind
//  - we do see the ast nodes for the llvm vendor
//  - we pick the right callees

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, col:21> col:5 implicit used also_before 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_1:0x[a-z0-9]*]] <<invalid sloc>> Implicit device={kind(cpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_2:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_3:0x[a-z0-9]*]] 'also_before[device={kind(cpu)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_3]] <col:1, line:8:1> line:6:1 also_before[device={kind(cpu)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_4:0x[a-z0-9]*]] <col:23, line:8:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_5:0x[a-z0-9]*]] <line:7:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_6:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:12:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:14:1> line:12:1 also_after[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:14:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:13:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_14:0x[a-z0-9]*]] prev [[ADDR_0]] <line:17:1, col:21> col:5 implicit used also_before 'int ({{.*}})'
// CHECK-NEXT: | |-OMPDeclareVariantAttr [[ADDR_15:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit device={kind(cpu)}
// CHECK-NEXT: | | `-DeclRefExpr [[ADDR_2]] <line:6:1> 'int ({{.*}})' Function [[ADDR_3]] 'also_before[device={kind(cpu)}]' 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_16:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(0): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_17:0x[a-z0-9]*]] <line:17:1> 'int ({{.*}})' Function [[ADDR_18:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_18]] <col:1, line:19:1> line:17:1 also_before[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_19:0x[a-z0-9]*]] <col:23, line:19:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_20:0x[a-z0-9]*]] <line:18:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_21:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] prev [[ADDR_7]] <line:22:1, line:24:1> line:22:5 used also_after 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_23:0x[a-z0-9]*]] <col:22, line:24:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_24:0x[a-z0-9]*]] <line:23:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_25:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_26:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:12:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: `-FunctionDecl [[ADDR_27:0x[a-z0-9]*]] <line:26:1, line:29:1> line:26:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_28:0x[a-z0-9]*]] <col:12, line:29:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_29:0x[a-z0-9]*]] <line:28:3, col:37>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_30:0x[a-z0-9]*]] <col:10, col:37> 'int' '+'
// CHECK-NEXT:         |-PseudoObjectExpr [[ADDR_31:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT:         | |-CallExpr [[ADDR_32:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT:         | | `-ImplicitCastExpr [[ADDR_33:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | |   `-DeclRefExpr [[ADDR_34:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_22]] 'also_after' 'int ({{.*}})'
// CHECK-NEXT:         | `-CallExpr [[ADDR_35:0x[a-z0-9]*]] <line:12:1, line:28:21> 'int'
// CHECK-NEXT:         |   `-ImplicitCastExpr [[ADDR_36:0x[a-z0-9]*]] <line:12:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |     `-DeclRefExpr [[ADDR_9]] <col:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT:         `-PseudoObjectExpr [[ADDR_37:0x[a-z0-9]*]] <line:28:25, col:37> 'int'
// CHECK-NEXT:           |-CallExpr [[ADDR_38:0x[a-z0-9]*]] <col:25, col:37> 'int'
// CHECK-NEXT:           | `-ImplicitCastExpr [[ADDR_39:0x[a-z0-9]*]] <col:25> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:           |   `-DeclRefExpr [[ADDR_40:0x[a-z0-9]*]] <col:25> 'int ({{.*}})' {{.*}}Function [[ADDR_14]] 'also_before' 'int ({{.*}})'
// CHECK-NEXT:           `-CallExpr [[ADDR_41:0x[a-z0-9]*]] <line:6:1, line:28:37> 'int'
// CHECK-NEXT:             `-ImplicitCastExpr [[ADDR_42:0x[a-z0-9]*]] <line:6:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:               `-DeclRefExpr [[ADDR_2]] <col:1> 'int ({{.*}})' Function [[ADDR_3]] 'also_before[device={kind(cpu)}]' 'int ({{.*}})'
