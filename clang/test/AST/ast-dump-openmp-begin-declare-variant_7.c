// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

int OK_1(void);

#pragma omp begin declare variant match(implementation={vendor(intel)})
int OK_1(void) {
  return 1;
}
int OK_2(void) {
  return 1;
}
int not_OK(void) {
  return 1;
}
int OK_3(void) {
  return 1;
}
#pragma omp end declare variant

int OK_3(void);

int test() {
  // Should cause an error due to not_OK()
  return OK_1() + not_OK() + OK_3();
}

// Make sure:
//  - we see a single error for `not_OK`
//  - we do not see errors for OK_{1,2,3}
//  FIXME: We actually do not see there error here.
//         This case is unlikely to happen in practise and hard to diagnose during SEMA.
//         We will issue an error during code generation instead. This is similar to the
//         diagnosis in other multi-versioning schemes.

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, col:14> col:5 used OK_1 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_1:0x[a-z0-9]*]] prev [[ADDR_0]] <line:8:1, col:14> col:5 implicit used OK_1 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_2:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(intel)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_3:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_4:0x[a-z0-9]*]] 'OK_1[implementation={vendor(intel)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_4]] <col:1, line:10:1> line:8:1 OK_1[implementation={vendor(intel)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_5:0x[a-z0-9]*]] <col:16, line:10:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_6:0x[a-z0-9]*]] <line:9:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_7:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |-FunctionDecl [[ADDR_8:0x[a-z0-9]*]] <line:11:1, col:14> col:5 implicit OK_2 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_9:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(intel)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_10:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_11:0x[a-z0-9]*]] 'OK_2[implementation={vendor(intel)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_11]] <col:1, line:13:1> line:11:1 OK_2[implementation={vendor(intel)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_12:0x[a-z0-9]*]] <col:16, line:13:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_13:0x[a-z0-9]*]] <line:12:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_14:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |-FunctionDecl [[ADDR_15:0x[a-z0-9]*]] <line:14:1, col:16> col:5 implicit used not_OK 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_16:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(intel)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_17:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_18:0x[a-z0-9]*]] 'not_OK[implementation={vendor(intel)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_18]] <col:1, line:16:1> line:14:1 not_OK[implementation={vendor(intel)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_19:0x[a-z0-9]*]] <col:18, line:16:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_20:0x[a-z0-9]*]] <line:15:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_21:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:17:1, col:14> col:5 implicit used OK_3 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_23:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(intel)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_24:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_25:0x[a-z0-9]*]] 'OK_3[implementation={vendor(intel)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_25]] <col:1, line:19:1> line:17:1 OK_3[implementation={vendor(intel)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_26:0x[a-z0-9]*]] <col:16, line:19:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_27:0x[a-z0-9]*]] <line:18:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |-FunctionDecl [[ADDR_29:0x[a-z0-9]*]] prev [[ADDR_22]] <line:22:1, col:14> col:5 used OK_3 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_30:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(intel)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_24]] <line:17:1> 'int ({{.*}})' Function [[ADDR_25]] 'OK_3[implementation={vendor(intel)}]' 'int ({{.*}})'
// CHECK-NEXT: `-FunctionDecl [[ADDR_31:0x[a-z0-9]*]] <line:24:1, line:27:1> line:24:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_32:0x[a-z0-9]*]] <col:12, line:27:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_33:0x[a-z0-9]*]] <line:26:3, col:35>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_34:0x[a-z0-9]*]] <col:10, col:35> 'int' '+'
// CHECK-NEXT:         |-BinaryOperator [[ADDR_35:0x[a-z0-9]*]] <col:10, col:26> 'int' '+'
// CHECK-NEXT:         | |-CallExpr [[ADDR_36:0x[a-z0-9]*]] <col:10, col:15> 'int'
// CHECK-NEXT:         | | `-ImplicitCastExpr [[ADDR_37:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | |   `-DeclRefExpr [[ADDR_38:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_1]] 'OK_1' 'int ({{.*}})'
// CHECK-NEXT:         | `-CallExpr [[ADDR_39:0x[a-z0-9]*]] <col:19, col:26> 'int'
// CHECK-NEXT:         |   `-ImplicitCastExpr [[ADDR_40:0x[a-z0-9]*]] <col:19> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |     `-DeclRefExpr [[ADDR_41:0x[a-z0-9]*]] <col:19> 'int ({{.*}})' {{.*}}Function [[ADDR_15]] 'not_OK' 'int ({{.*}})'
// CHECK-NEXT:         `-CallExpr [[ADDR_42:0x[a-z0-9]*]] <col:30, col:35> 'int'
// CHECK-NEXT:           `-ImplicitCastExpr [[ADDR_43:0x[a-z0-9]*]] <col:30> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:             `-DeclRefExpr [[ADDR_44:0x[a-z0-9]*]] <col:30> 'int ({{.*}})' {{.*}}Function [[ADDR_29]] 'OK_3' 'int ({{.*}})'
