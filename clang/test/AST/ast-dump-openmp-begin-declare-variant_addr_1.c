// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(implementation={vendor(llvm)})
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

int test(int (*fd)(void)) {
  return fd();
}
int main(void) {
  // Should return 0.
  return test(also_after) +
         test(also_before) +
         test(&also_after) +
         test(&also_before);
}

// Make sure:
//  - we see the specialization in the AST
//  - we pick the right callees

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:13:1> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:12:1> line:10:1 also_after[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:12:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:11:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |-FunctionDecl [[ADDR_6]] <line:13:1, line:15:1> line:13:1 also_before[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_14:0x[a-z0-9]*]] <col:23, line:15:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_15:0x[a-z0-9]*]] <line:14:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: |-FunctionDecl [[ADDR_17:0x[a-z0-9]*]] prev [[ADDR_7]] <line:18:1, line:20:1> line:18:5 used also_after 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:22, line:20:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_19:0x[a-z0-9]*]] <line:19:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_20:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_21:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:10:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:22:1, line:24:1> line:22:5 used test 'int (int (*)({{.*}}))'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_23:0x[a-z0-9]*]] <col:10, col:24> col:16 used fd 'int (*)({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_24:0x[a-z0-9]*]] <col:27, line:24:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_25:0x[a-z0-9]*]] <line:23:3, col:13>
// CHECK-NEXT: |     `-CallExpr [[ADDR_26:0x[a-z0-9]*]] <col:10, col:13> 'int'
// CHECK-NEXT: |       `-ImplicitCastExpr [[ADDR_27:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' {{.*}}ParmVar [[ADDR_23]] 'fd' 'int (*)({{.*}})'
// CHECK-NEXT: `-FunctionDecl [[ADDR_29:0x[a-z0-9]*]] <line:25:1, line:31:1> line:25:5 main 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_30:0x[a-z0-9]*]] <col:16, line:31:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_31:0x[a-z0-9]*]] <line:27:3, line:30:27>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_32:0x[a-z0-9]*]] <line:27:10, line:30:27> 'int' '+'
// CHECK-NEXT:         |-BinaryOperator [[ADDR_33:0x[a-z0-9]*]] <line:27:10, line:29:26> 'int' '+'
// CHECK-NEXT:         | |-BinaryOperator [[ADDR_34:0x[a-z0-9]*]] <line:27:10, line:28:26> 'int' '+'
// CHECK-NEXT:         | | |-CallExpr [[ADDR_35:0x[a-z0-9]*]] <line:27:10, col:25> 'int'
// CHECK-NEXT:         | | | |-ImplicitCastExpr [[ADDR_36:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | `-DeclRefExpr [[ADDR_37:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' {{.*}}Function [[ADDR_22]] 'test' 'int (int (*)({{.*}}))'
// CHECK-NEXT:         | | | `-ImplicitCastExpr [[ADDR_38:0x[a-z0-9]*]] <col:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | |   `-DeclRefExpr [[ADDR_39:0x[a-z0-9]*]] <col:15> 'int ({{.*}})' {{.*}}Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// CHECK-NEXT:         | | `-CallExpr [[ADDR_40:0x[a-z0-9]*]] <line:28:10, col:26> 'int'
// CHECK-NEXT:         | |   |-ImplicitCastExpr [[ADDR_41:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// CHECK-NEXT:         | |   | `-DeclRefExpr [[ADDR_42:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' {{.*}}Function [[ADDR_22]] 'test' 'int (int (*)({{.*}}))'
// CHECK-NEXT:         | |   `-ImplicitCastExpr [[ADDR_43:0x[a-z0-9]*]] <col:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | |     `-DeclRefExpr [[ADDR_44:0x[a-z0-9]*]] <col:15> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CHECK-NEXT:         | `-CallExpr [[ADDR_45:0x[a-z0-9]*]] <line:29:10, col:26> 'int'
// CHECK-NEXT:         |   |-ImplicitCastExpr [[ADDR_46:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// CHECK-NEXT:         |   | `-DeclRefExpr [[ADDR_47:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' {{.*}}Function [[ADDR_22]] 'test' 'int (int (*)({{.*}}))'
// CHECK-NEXT:         |   `-UnaryOperator [[ADDR_48:0x[a-z0-9]*]] <col:15, col:16> 'int (*)({{.*}})' prefix '&' cannot overflow
// CHECK-NEXT:         |     `-DeclRefExpr [[ADDR_49:0x[a-z0-9]*]] <col:16> 'int ({{.*}})' {{.*}}Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// CHECK-NEXT:         `-CallExpr [[ADDR_50:0x[a-z0-9]*]] <line:30:10, col:27> 'int'
// CHECK-NEXT:           |-ImplicitCastExpr [[ADDR_51:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// CHECK-NEXT:           | `-DeclRefExpr [[ADDR_52:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' {{.*}}Function [[ADDR_22]] 'test' 'int (int (*)({{.*}}))'
// CHECK-NEXT:           `-UnaryOperator [[ADDR_53:0x[a-z0-9]*]] <col:15, col:16> 'int (*)({{.*}})' prefix '&' cannot overflow
// CHECK-NEXT:             `-DeclRefExpr [[ADDR_54:0x[a-z0-9]*]] <col:16> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
