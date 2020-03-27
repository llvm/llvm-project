// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s --check-prefix=C
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s --check-prefix=CXX
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
int main() {
  // Should return 0.
  return test(also_after) +
         test(also_before) +
         test(&also_after) +
         test(&also_before);
}

// Make sure:
//  - we see the specialization in the AST
//  - we pick the right callees

// CXX:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_4:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_5:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_6:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_7:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_7]] <col:1, line:12:1> line:10:1 also_after[implementation={vendor(llvm)}] 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_8:0x[a-z0-9]*]] <col:22, line:12:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_9:0x[a-z0-9]*]] <line:11:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_10:0x[a-z0-9]*]] <col:10> 'int' 1
// CXX-NEXT: |-FunctionDecl [[ADDR_11:0x[a-z0-9]*]] prev [[ADDR_0]] <line:13:1, col:21> col:5 implicit used also_before 'int ({{.*}})'
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_12:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_13:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_14:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_14]] <col:1, line:15:1> line:13:1 also_before[implementation={vendor(llvm)}] 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_15:0x[a-z0-9]*]] <col:23, line:15:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_16:0x[a-z0-9]*]] <line:14:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_17:0x[a-z0-9]*]] <col:10> 'int' 2
// CXX-NEXT: |-FunctionDecl [[ADDR_18:0x[a-z0-9]*]] prev [[ADDR_4]] <line:18:1, line:20:1> line:18:5 used also_after 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_19:0x[a-z0-9]*]] <col:22, line:20:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_20:0x[a-z0-9]*]] <line:19:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_21:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_22:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_6]] <line:10:1> 'int ({{.*}})' Function [[ADDR_7]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_23:0x[a-z0-9]*]] <line:22:1, line:24:1> line:22:5 used test 'int (int (*)({{.*}}))'
// CXX-NEXT: | |-ParmVarDecl [[ADDR_24:0x[a-z0-9]*]] <col:10, col:24> col:16 used fd 'int (*)({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_25:0x[a-z0-9]*]] <col:27, line:24:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_26:0x[a-z0-9]*]] <line:23:3, col:13>
// CXX-NEXT: |     `-CallExpr [[ADDR_27:0x[a-z0-9]*]] <col:10, col:13> 'int'
// CXX-NEXT: |       `-ImplicitCastExpr [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <LValueToRValue>
// CXX-NEXT: |         `-DeclRefExpr [[ADDR_29:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' {{.*}}ParmVar [[ADDR_24]] 'fd' 'int (*)({{.*}})'
// CXX-NEXT: `-FunctionDecl [[ADDR_30:0x[a-z0-9]*]] <line:25:1, line:31:1> line:25:5 main 'int ({{.*}})'
// CXX-NEXT:   `-CompoundStmt [[ADDR_31:0x[a-z0-9]*]] <col:12, line:31:1>
// CXX-NEXT:     `-ReturnStmt [[ADDR_32:0x[a-z0-9]*]] <line:27:3, line:30:27>
// CXX-NEXT:       `-BinaryOperator [[ADDR_33:0x[a-z0-9]*]] <line:27:10, line:30:27> 'int' '+'
// CXX-NEXT:         |-BinaryOperator [[ADDR_34:0x[a-z0-9]*]] <line:27:10, line:29:26> 'int' '+'
// CXX-NEXT:         | |-BinaryOperator [[ADDR_35:0x[a-z0-9]*]] <line:27:10, line:28:26> 'int' '+'
// CXX-NEXT:         | | |-CallExpr [[ADDR_36:0x[a-z0-9]*]] <line:27:10, col:25> 'int'
// CXX-NEXT:         | | | |-ImplicitCastExpr [[ADDR_37:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// CXX-NEXT:         | | | | `-DeclRefExpr [[ADDR_38:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' {{.*}}Function [[ADDR_23]] 'test' 'int (int (*)({{.*}}))'
// CXX-NEXT:         | | | `-ImplicitCastExpr [[ADDR_39:0x[a-z0-9]*]] <col:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | | |   `-DeclRefExpr [[ADDR_40:0x[a-z0-9]*]] <col:15> 'int ({{.*}})' {{.*}}Function [[ADDR_18]] 'also_after' 'int ({{.*}})'
// CXX-NEXT:         | | `-CallExpr [[ADDR_41:0x[a-z0-9]*]] <line:28:10, col:26> 'int'
// CXX-NEXT:         | |   |-ImplicitCastExpr [[ADDR_42:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// CXX-NEXT:         | |   | `-DeclRefExpr [[ADDR_43:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' {{.*}}Function [[ADDR_23]] 'test' 'int (int (*)({{.*}}))'
// CXX-NEXT:         | |   `-ImplicitCastExpr [[ADDR_44:0x[a-z0-9]*]] <col:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | |     `-DeclRefExpr [[ADDR_45:0x[a-z0-9]*]] <col:15> 'int ({{.*}})' {{.*}}Function [[ADDR_11]] 'also_before' 'int ({{.*}})'
// CXX-NEXT:         | `-CallExpr [[ADDR_46:0x[a-z0-9]*]] <line:29:10, col:26> 'int'
// CXX-NEXT:         |   |-ImplicitCastExpr [[ADDR_47:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// CXX-NEXT:         |   | `-DeclRefExpr [[ADDR_48:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' {{.*}}Function [[ADDR_23]] 'test' 'int (int (*)({{.*}}))'
// CXX-NEXT:         |   `-UnaryOperator [[ADDR_49:0x[a-z0-9]*]] <col:15, col:16> 'int (*)({{.*}})' prefix '&' cannot overflow
// CXX-NEXT:         |     `-DeclRefExpr [[ADDR_50:0x[a-z0-9]*]] <col:16> 'int ({{.*}})' {{.*}}Function [[ADDR_18]] 'also_after' 'int ({{.*}})'
// CXX-NEXT:         `-CallExpr [[ADDR_51:0x[a-z0-9]*]] <line:30:10, col:27> 'int'
// CXX-NEXT:           |-ImplicitCastExpr [[ADDR_52:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// CXX-NEXT:           | `-DeclRefExpr [[ADDR_53:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' {{.*}}Function [[ADDR_23]] 'test' 'int (int (*)({{.*}}))'
// CXX-NEXT:           `-UnaryOperator [[ADDR_54:0x[a-z0-9]*]] <col:15, col:16> 'int (*)({{.*}})' prefix '&' cannot overflow
// CXX-NEXT:             `-DeclRefExpr [[ADDR_55:0x[a-z0-9]*]] <col:16> 'int ({{.*}})' {{.*}}Function [[ADDR_11]] 'also_before' 'int ({{.*}})'

// C:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// C-NEXT: | `-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// C-NEXT: |     `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: |-FunctionDecl [[ADDR_4:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_5:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_6:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_7:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_7]] <col:1, line:12:1> line:10:1 also_after[implementation={vendor(llvm)}] 'int ({{.*}})'
// C-NEXT: | `-CompoundStmt [[ADDR_8:0x[a-z0-9]*]] <col:22, line:12:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_9:0x[a-z0-9]*]] <line:11:3, col:10>
// C-NEXT: |     `-IntegerLiteral [[ADDR_10:0x[a-z0-9]*]] <col:10> 'int' 1
// C-NEXT: |-FunctionDecl [[ADDR_11:0x[a-z0-9]*]] prev [[ADDR_0]] <line:13:1, col:21> col:5 implicit used also_before 'int ({{.*}})'
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_12:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_13:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_14:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_14]] <col:1, line:15:1> line:13:1 also_before[implementation={vendor(llvm)}] 'int ({{.*}})'
// C-NEXT: | `-CompoundStmt [[ADDR_15:0x[a-z0-9]*]] <col:23, line:15:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_16:0x[a-z0-9]*]] <line:14:3, col:10>
// C-NEXT: |     `-IntegerLiteral [[ADDR_17:0x[a-z0-9]*]] <col:10> 'int' 2
// C-NEXT: |-FunctionDecl [[ADDR_18:0x[a-z0-9]*]] prev [[ADDR_4]] <line:18:1, line:20:1> line:18:5 used also_after 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_19:0x[a-z0-9]*]] <col:22, line:20:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_20:0x[a-z0-9]*]] <line:19:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_21:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_22:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_6]] <line:10:1> 'int ({{.*}})' Function [[ADDR_7]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_23:0x[a-z0-9]*]] <line:22:1, line:24:1> line:22:5 used test 'int (int (*)({{.*}}))'
// C-NEXT: | |-ParmVarDecl [[ADDR_24:0x[a-z0-9]*]] <col:10, col:24> col:16 used fd 'int (*)({{.*}})'
// C-NEXT: | `-CompoundStmt [[ADDR_25:0x[a-z0-9]*]] <col:27, line:24:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_26:0x[a-z0-9]*]] <line:23:3, col:13>
// C-NEXT: |     `-CallExpr [[ADDR_27:0x[a-z0-9]*]] <col:10, col:13> 'int'
// C-NEXT: |       `-ImplicitCastExpr [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <LValueToRValue>
// C-NEXT: |         `-DeclRefExpr [[ADDR_29:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' {{.*}}ParmVar [[ADDR_24]] 'fd' 'int (*)({{.*}})'
// C-NEXT: `-FunctionDecl [[ADDR_30:0x[a-z0-9]*]] <line:25:1, line:31:1> line:25:5 main 'int ({{.*}})'
// C-NEXT:   `-CompoundStmt [[ADDR_31:0x[a-z0-9]*]] <col:12, line:31:1>
// C-NEXT:     `-ReturnStmt [[ADDR_32:0x[a-z0-9]*]] <line:27:3, line:30:27>
// C-NEXT:       `-BinaryOperator [[ADDR_33:0x[a-z0-9]*]] <line:27:10, line:30:27> 'int' '+'
// C-NEXT:         |-BinaryOperator [[ADDR_34:0x[a-z0-9]*]] <line:27:10, line:29:26> 'int' '+'
// C-NEXT:         | |-BinaryOperator [[ADDR_35:0x[a-z0-9]*]] <line:27:10, line:28:26> 'int' '+'
// C-NEXT:         | | |-CallExpr [[ADDR_36:0x[a-z0-9]*]] <line:27:10, col:25> 'int'
// C-NEXT:         | | | |-ImplicitCastExpr [[ADDR_37:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// C-NEXT:         | | | | `-DeclRefExpr [[ADDR_38:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' Function [[ADDR_23]] 'test' 'int (int (*)({{.*}}))'
// C-NEXT:         | | | `-ImplicitCastExpr [[ADDR_39:0x[a-z0-9]*]] <col:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | | |   `-DeclRefExpr [[ADDR_40:0x[a-z0-9]*]] <col:15> 'int ({{.*}})' Function [[ADDR_18]] 'also_after' 'int ({{.*}})'
// C-NEXT:         | | `-CallExpr [[ADDR_41:0x[a-z0-9]*]] <line:28:10, col:26> 'int'
// C-NEXT:         | |   |-ImplicitCastExpr [[ADDR_42:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// C-NEXT:         | |   | `-DeclRefExpr [[ADDR_43:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' Function [[ADDR_23]] 'test' 'int (int (*)({{.*}}))'
// C-NEXT:         | |   `-ImplicitCastExpr [[ADDR_44:0x[a-z0-9]*]] <col:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | |     `-DeclRefExpr [[ADDR_45:0x[a-z0-9]*]] <col:15> 'int ({{.*}})' Function [[ADDR_11]] 'also_before' 'int ({{.*}})'
// C-NEXT:         | `-CallExpr [[ADDR_46:0x[a-z0-9]*]] <line:29:10, col:26> 'int'
// C-NEXT:         |   |-ImplicitCastExpr [[ADDR_47:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// C-NEXT:         |   | `-DeclRefExpr [[ADDR_48:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' Function [[ADDR_23]] 'test' 'int (int (*)({{.*}}))'
// C-NEXT:         |   `-UnaryOperator [[ADDR_49:0x[a-z0-9]*]] <col:15, col:16> 'int (*)({{.*}})' prefix '&' cannot overflow
// C-NEXT:         |     `-DeclRefExpr [[ADDR_50:0x[a-z0-9]*]] <col:16> 'int ({{.*}})' Function [[ADDR_18]] 'also_after' 'int ({{.*}})'
// C-NEXT:         `-CallExpr [[ADDR_51:0x[a-z0-9]*]] <line:30:10, col:27> 'int'
// C-NEXT:           |-ImplicitCastExpr [[ADDR_52:0x[a-z0-9]*]] <col:10> 'int (*)(int (*)({{.*}}))' <FunctionToPointerDecay>
// C-NEXT:           | `-DeclRefExpr [[ADDR_53:0x[a-z0-9]*]] <col:10> 'int (int (*)({{.*}}))' Function [[ADDR_23]] 'test' 'int (int (*)({{.*}}))'
// C-NEXT:           `-UnaryOperator [[ADDR_54:0x[a-z0-9]*]] <col:15, col:16> 'int (*)({{.*}})' prefix '&' cannot overflow
// C-NEXT:             `-DeclRefExpr [[ADDR_55:0x[a-z0-9]*]] <col:16> 'int ({{.*}})' Function [[ADDR_11]] 'also_before' 'int ({{.*}})'
