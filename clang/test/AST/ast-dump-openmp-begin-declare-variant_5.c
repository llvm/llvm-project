// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s --check-prefix=C
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s --check-prefix=CXX
// expected-no-diagnostics

int also_before(void) {
  return 1;
}

#pragma omp begin declare variant match(implementation={vendor(amd)})
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

int main(void) {
  // Should return 0.
  return (also_after)() +
         (also_before)() +
         (&also_after)() +
         (&also_before)();
}

// Make sure:
//  - we see the specialization in the AST
//  - we pick the right callees

// C:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 1
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(amd)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:13:1> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[implementation={vendor(amd)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(amd)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[implementation={vendor(amd)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:12:1> line:10:1 also_after[implementation={vendor(amd)}] 'int ({{.*}})'
// C-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:12:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:11:3, col:10>
// C-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: |-FunctionDecl [[ADDR_6]] <line:13:1, line:15:1> line:13:1 also_before[implementation={vendor(amd)}] 'int ({{.*}})'
// C-NEXT: | `-CompoundStmt [[ADDR_14:0x[a-z0-9]*]] <col:23, line:15:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_15:0x[a-z0-9]*]] <line:14:3, col:10>
// C-NEXT: |     `-IntegerLiteral [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: |-FunctionDecl [[ADDR_17:0x[a-z0-9]*]] prev [[ADDR_7]] <line:18:1, line:20:1> line:18:5 used also_after 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:22, line:20:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_19:0x[a-z0-9]*]] <line:19:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_20:0x[a-z0-9]*]] <col:10> 'int' 2
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_21:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(amd)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:10:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(amd)}]' 'int ({{.*}})'
// C-NEXT: `-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:22:1, line:28:1> line:22:5 main 'int ({{.*}})'
// C-NEXT:   `-CompoundStmt [[ADDR_23:0x[a-z0-9]*]] <col:16, line:28:1>
// C-NEXT:     `-ReturnStmt [[ADDR_24:0x[a-z0-9]*]] <line:24:3, line:27:25>
// C-NEXT:       `-BinaryOperator [[ADDR_25:0x[a-z0-9]*]] <line:24:10, line:27:25> 'int' '+'
// C-NEXT:         |-BinaryOperator [[ADDR_26:0x[a-z0-9]*]] <line:24:10, line:26:24> 'int' '+'
// C-NEXT:         | |-BinaryOperator [[ADDR_27:0x[a-z0-9]*]] <line:24:10, line:25:24> 'int' '+'
// C-NEXT:         | | |-PseudoObjectExpr [[ADDR_28:0x[a-z0-9]*]] <line:24:10, col:23> 'int'
// C-NEXT:         | | | |-CallExpr [[ADDR_29:0x[a-z0-9]*]] <col:10, col:23> 'int'
// C-NEXT:         | | | | `-ImplicitCastExpr [[ADDR_30:0x[a-z0-9]*]] <col:10, col:21> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | | | |   `-ParenExpr [[ADDR_31:0x[a-z0-9]*]] <col:10, col:21> 'int ({{.*}})'
// C-NEXT:         | | | |     `-DeclRefExpr [[ADDR_32:0x[a-z0-9]*]] <col:11> 'int ({{.*}})' Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// C-NEXT:         | | | `-CallExpr [[ADDR_33:0x[a-z0-9]*]] <line:10:1, line:24:23> 'int'
// C-NEXT:         | | |   `-ImplicitCastExpr [[ADDR_34:0x[a-z0-9]*]] <line:10:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | | |     `-DeclRefExpr [[ADDR_9]] <col:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(amd)}]' 'int ({{.*}})'
// C-NEXT:         | | `-PseudoObjectExpr [[ADDR_35:0x[a-z0-9]*]] <line:25:10, col:24> 'int'
// C-NEXT:         | |   |-CallExpr [[ADDR_36:0x[a-z0-9]*]] <col:10, col:24> 'int'
// C-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_37:0x[a-z0-9]*]] <col:10, col:22> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | |   |   `-ParenExpr [[ADDR_38:0x[a-z0-9]*]] <col:10, col:22> 'int ({{.*}})'
// C-NEXT:         | |   |     `-DeclRefExpr [[ADDR_39:0x[a-z0-9]*]] <col:11> 'int ({{.*}})' Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// C-NEXT:         | |   `-CallExpr [[ADDR_40:0x[a-z0-9]*]] <line:13:1, line:25:24> 'int'
// C-NEXT:         | |     `-ImplicitCastExpr [[ADDR_41:0x[a-z0-9]*]] <line:13:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | |       `-DeclRefExpr [[ADDR_5]] <col:1> 'int ({{.*}})' Function [[ADDR_6]] 'also_before[implementation={vendor(amd)}]' 'int ({{.*}})'
// C-NEXT:         | `-PseudoObjectExpr [[ADDR_42:0x[a-z0-9]*]] <line:26:10, col:24> 'int'
// C-NEXT:         |   |-CallExpr [[ADDR_43:0x[a-z0-9]*]] <col:10, col:24> 'int'
// C-NEXT:         |   | `-ParenExpr [[ADDR_44:0x[a-z0-9]*]] <col:10, col:22> 'int (*)({{.*}})'
// C-NEXT:         |   |   `-UnaryOperator [[ADDR_45:0x[a-z0-9]*]] <col:11, col:12> 'int (*)({{.*}})' prefix '&' cannot overflow
// C-NEXT:         |   |     `-DeclRefExpr [[ADDR_46:0x[a-z0-9]*]] <col:12> 'int ({{.*}})' Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// C-NEXT:         |   `-CallExpr [[ADDR_47:0x[a-z0-9]*]] <line:10:1, line:26:24> 'int'
// C-NEXT:         |     `-ImplicitCastExpr [[ADDR_48:0x[a-z0-9]*]] <line:10:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         |       `-DeclRefExpr [[ADDR_9]] <col:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(amd)}]' 'int ({{.*}})'
// C-NEXT:         `-PseudoObjectExpr [[ADDR_49:0x[a-z0-9]*]] <line:27:10, col:25> 'int'
// C-NEXT:           |-CallExpr [[ADDR_50:0x[a-z0-9]*]] <col:10, col:25> 'int'
// C-NEXT:           | `-ParenExpr [[ADDR_51:0x[a-z0-9]*]] <col:10, col:23> 'int (*)({{.*}})'
// C-NEXT:           |   `-UnaryOperator [[ADDR_52:0x[a-z0-9]*]] <col:11, col:12> 'int (*)({{.*}})' prefix '&' cannot overflow
// C-NEXT:           |     `-DeclRefExpr [[ADDR_53:0x[a-z0-9]*]] <col:12> 'int ({{.*}})' Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// C-NEXT:           `-CallExpr [[ADDR_54:0x[a-z0-9]*]] <line:13:1, line:27:25> 'int'
// C-NEXT:             `-ImplicitCastExpr [[ADDR_55:0x[a-z0-9]*]] <line:13:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:               `-DeclRefExpr [[ADDR_5]] <col:1> 'int ({{.*}})' Function [[ADDR_6]] 'also_before[implementation={vendor(amd)}]' 'int ({{.*}})'

// CXX:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 1
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(amd)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:13:1> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[implementation={vendor(amd)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(amd)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[implementation={vendor(amd)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:12:1> line:10:1 also_after[implementation={vendor(amd)}] 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:12:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:11:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_6]] <line:13:1, line:15:1> line:13:1 also_before[implementation={vendor(amd)}] 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_14:0x[a-z0-9]*]] <col:23, line:15:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_15:0x[a-z0-9]*]] <line:14:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_17:0x[a-z0-9]*]] prev [[ADDR_7]] <line:18:1, line:20:1> line:18:5 used also_after 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:22, line:20:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_19:0x[a-z0-9]*]] <line:19:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_20:0x[a-z0-9]*]] <col:10> 'int' 2
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_21:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(amd)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:10:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(amd)}]' 'int ({{.*}})'
// CXX-NEXT: `-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:22:1, line:28:1> line:22:5 main 'int ({{.*}})'
// CXX-NEXT:   `-CompoundStmt [[ADDR_23:0x[a-z0-9]*]] <col:16, line:28:1>
// CXX-NEXT:     `-ReturnStmt [[ADDR_24:0x[a-z0-9]*]] <line:24:3, line:27:25>
// CXX-NEXT:       `-BinaryOperator [[ADDR_25:0x[a-z0-9]*]] <line:24:10, line:27:25> 'int' '+'
// CXX-NEXT:         |-BinaryOperator [[ADDR_26:0x[a-z0-9]*]] <line:24:10, line:26:24> 'int' '+'
// CXX-NEXT:         | |-BinaryOperator [[ADDR_27:0x[a-z0-9]*]] <line:24:10, line:25:24> 'int' '+'
// CXX-NEXT:         | | |-PseudoObjectExpr [[ADDR_28:0x[a-z0-9]*]] <line:24:10, col:23> 'int'
// CXX-NEXT:         | | | |-CallExpr [[ADDR_29:0x[a-z0-9]*]] <col:10, col:23> 'int'
// CXX-NEXT:         | | | | `-ImplicitCastExpr [[ADDR_30:0x[a-z0-9]*]] <col:10, col:21> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | | | |   `-ParenExpr [[ADDR_31:0x[a-z0-9]*]] <col:10, col:21> 'int ({{.*}})' lvalue
// CXX-NEXT:         | | | |     `-DeclRefExpr [[ADDR_32:0x[a-z0-9]*]] <col:11> 'int ({{.*}})' {{.*}}Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// CXX-NEXT:         | | | `-CallExpr [[ADDR_33:0x[a-z0-9]*]] <line:10:1, line:24:23> 'int'
// CXX-NEXT:         | | |   `-ImplicitCastExpr [[ADDR_34:0x[a-z0-9]*]] <line:10:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | | |     `-DeclRefExpr [[ADDR_9]] <col:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(amd)}]' 'int ({{.*}})'
// CXX-NEXT:         | | `-PseudoObjectExpr [[ADDR_35:0x[a-z0-9]*]] <line:25:10, col:24> 'int'
// CXX-NEXT:         | |   |-CallExpr [[ADDR_36:0x[a-z0-9]*]] <col:10, col:24> 'int'
// CXX-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_37:0x[a-z0-9]*]] <col:10, col:22> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | |   |   `-ParenExpr [[ADDR_38:0x[a-z0-9]*]] <col:10, col:22> 'int ({{.*}})' lvalue
// CXX-NEXT:         | |   |     `-DeclRefExpr [[ADDR_39:0x[a-z0-9]*]] <col:11> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CXX-NEXT:         | |   `-CallExpr [[ADDR_40:0x[a-z0-9]*]] <line:13:1, line:25:24> 'int'
// CXX-NEXT:         | |     `-ImplicitCastExpr [[ADDR_41:0x[a-z0-9]*]] <line:13:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | |       `-DeclRefExpr [[ADDR_5]] <col:1> 'int ({{.*}})' Function [[ADDR_6]] 'also_before[implementation={vendor(amd)}]' 'int ({{.*}})'
// CXX-NEXT:         | `-PseudoObjectExpr [[ADDR_42:0x[a-z0-9]*]] <line:26:10, col:24> 'int'
// CXX-NEXT:         |   |-CallExpr [[ADDR_43:0x[a-z0-9]*]] <col:10, col:24> 'int'
// CXX-NEXT:         |   | `-ParenExpr [[ADDR_44:0x[a-z0-9]*]] <col:10, col:22> 'int (*)({{.*}})'
// CXX-NEXT:         |   |   `-UnaryOperator [[ADDR_45:0x[a-z0-9]*]] <col:11, col:12> 'int (*)({{.*}})' prefix '&' cannot overflow
// CXX-NEXT:         |   |     `-DeclRefExpr [[ADDR_46:0x[a-z0-9]*]] <col:12> 'int ({{.*}})' {{.*}}Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// CXX-NEXT:         |   `-CallExpr [[ADDR_47:0x[a-z0-9]*]] <line:10:1, line:26:24> 'int'
// CXX-NEXT:         |     `-ImplicitCastExpr [[ADDR_48:0x[a-z0-9]*]] <line:10:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         |       `-DeclRefExpr [[ADDR_9]] <col:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(amd)}]' 'int ({{.*}})'
// CXX-NEXT:         `-PseudoObjectExpr [[ADDR_49:0x[a-z0-9]*]] <line:27:10, col:25> 'int'
// CXX-NEXT:           |-CallExpr [[ADDR_50:0x[a-z0-9]*]] <col:10, col:25> 'int'
// CXX-NEXT:           | `-ParenExpr [[ADDR_51:0x[a-z0-9]*]] <col:10, col:23> 'int (*)({{.*}})'
// CXX-NEXT:           |   `-UnaryOperator [[ADDR_52:0x[a-z0-9]*]] <col:11, col:12> 'int (*)({{.*}})' prefix '&' cannot overflow
// CXX-NEXT:           |     `-DeclRefExpr [[ADDR_53:0x[a-z0-9]*]] <col:12> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CXX-NEXT:           `-CallExpr [[ADDR_54:0x[a-z0-9]*]] <line:13:1, line:27:25> 'int'
// CXX-NEXT:             `-ImplicitCastExpr [[ADDR_55:0x[a-z0-9]*]] <line:13:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:               `-DeclRefExpr [[ADDR_5]] <col:1> 'int ({{.*}})' Function [[ADDR_6]] 'also_before[implementation={vendor(amd)}]' 'int ({{.*}})'
