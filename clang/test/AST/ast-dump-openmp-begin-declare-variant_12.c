// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s --check-prefix=C
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s --check-prefix=CXX
// expected-no-diagnostics

#ifdef __cplusplus
#define OVERLOADABLE
#else
#define OVERLOADABLE __attribute__((overloadable))
#endif

OVERLOADABLE
int also_before(void) {
  return 1;
}
OVERLOADABLE
int also_before(int i) {
  return 2;
}
OVERLOADABLE
int also_before(float f) {
  return 0;
}
OVERLOADABLE
int also_before(double d) {
  return 3;
}
OVERLOADABLE
int also_before(long l) {
  return 4;
}

#pragma omp begin declare variant match(implementation = {vendor(llvm)})
OVERLOADABLE
int also_before(void) {
  return 0;
}
OVERLOADABLE
int also_before(int i) {
  return 0;
}
// No float!
OVERLOADABLE
int also_before(double d) {
  return 0;
}
OVERLOADABLE
int also_before(long l) {
  return 0;
}
#pragma omp end declare variant


int main(void) {
  // Should return 0.
  return also_before() + also_before(1) + also_before(2.0f) + also_before(3.0) + also_before(4L);
}

// Make sure:
//  - we see the specialization in the AST
//  - we pick the right callees

// C:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:14:1> line:12:5 used also_before 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:14:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:13:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 1
// C-NEXT: | |-OverloadableAttr [[ADDR_4:0x[a-z0-9]*]] <line:8:37>
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_5:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_6:0x[a-z0-9]*]] <col:22> 'int ({{.*}})' Function [[ADDR_7:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_8:0x[a-z0-9]*]] <col:22, line:18:1> line:16:5 used also_before 'int (int)'
// C-NEXT: | |-ParmVarDecl [[ADDR_9:0x[a-z0-9]*]] <col:17, col:21> col:21 i 'int'
// C-NEXT: | |-CompoundStmt [[ADDR_10:0x[a-z0-9]*]] <col:24, line:18:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_11:0x[a-z0-9]*]] <line:17:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_12:0x[a-z0-9]*]] <col:10> 'int' 2
// C-NEXT: | |-OverloadableAttr [[ADDR_13:0x[a-z0-9]*]] <line:8:37>
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_14:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_15:0x[a-z0-9]*]] <col:22> 'int (int)' Function [[ADDR_16:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int (int)'
// C-NEXT: |-FunctionDecl [[ADDR_17:0x[a-z0-9]*]] <col:22, line:22:1> line:20:5 used also_before 'int (float)'
// C-NEXT: | |-ParmVarDecl [[ADDR_18:0x[a-z0-9]*]] <col:17, col:23> col:23 f 'float'
// C-NEXT: | |-CompoundStmt [[ADDR_19:0x[a-z0-9]*]] <col:26, line:22:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_20:0x[a-z0-9]*]] <line:21:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_21:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-OverloadableAttr [[ADDR_22:0x[a-z0-9]*]] <line:8:37>
// C-NEXT: |-FunctionDecl [[ADDR_23:0x[a-z0-9]*]] <col:22, line:26:1> line:24:5 used also_before 'int (double)'
// C-NEXT: | |-ParmVarDecl [[ADDR_24:0x[a-z0-9]*]] <col:17, col:24> col:24 d 'double'
// C-NEXT: | |-CompoundStmt [[ADDR_25:0x[a-z0-9]*]] <col:27, line:26:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_26:0x[a-z0-9]*]] <line:25:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_27:0x[a-z0-9]*]] <col:10> 'int' 3
// C-NEXT: | |-OverloadableAttr [[ADDR_28:0x[a-z0-9]*]] <line:8:37>
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_29:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_30:0x[a-z0-9]*]] <col:22> 'int (double)' Function [[ADDR_31:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int (double)'
// C-NEXT: |-FunctionDecl [[ADDR_32:0x[a-z0-9]*]] <col:22, line:30:1> line:28:5 used also_before 'int (long)'
// C-NEXT: | |-ParmVarDecl [[ADDR_33:0x[a-z0-9]*]] <col:17, col:22> col:22 l 'long'
// C-NEXT: | |-CompoundStmt [[ADDR_34:0x[a-z0-9]*]] <col:25, line:30:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_35:0x[a-z0-9]*]] <line:29:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_36:0x[a-z0-9]*]] <col:10> 'int' 4
// C-NEXT: | |-OverloadableAttr [[ADDR_37:0x[a-z0-9]*]] <line:8:37>
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_38:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_39:0x[a-z0-9]*]] <col:22> 'int (long)' Function [[ADDR_40:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int (long)'
// C-NEXT: |-FunctionDecl [[ADDR_7]] <col:22, line:36:1> line:8:22 also_before[implementation={vendor(llvm)}] 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_41:0x[a-z0-9]*]] <line:34:23, line:36:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_42:0x[a-z0-9]*]] <line:35:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_43:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-OverloadableAttr [[ADDR_44:0x[a-z0-9]*]] <line:8:37>
// C-NEXT: |-FunctionDecl [[ADDR_16]] <col:22, line:40:1> line:8:22 also_before[implementation={vendor(llvm)}] 'int (int)'
// C-NEXT: | |-ParmVarDecl [[ADDR_45:0x[a-z0-9]*]] <line:38:17, col:21> col:21 i 'int'
// C-NEXT: | |-CompoundStmt [[ADDR_46:0x[a-z0-9]*]] <col:24, line:40:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_47:0x[a-z0-9]*]] <line:39:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_48:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-OverloadableAttr [[ADDR_49:0x[a-z0-9]*]] <line:8:37>
// C-NEXT: |-FunctionDecl [[ADDR_31]] <col:22, line:45:1> line:8:22 also_before[implementation={vendor(llvm)}] 'int (double)'
// C-NEXT: | |-ParmVarDecl [[ADDR_50:0x[a-z0-9]*]] <line:43:17, col:24> col:24 d 'double'
// C-NEXT: | |-CompoundStmt [[ADDR_51:0x[a-z0-9]*]] <col:27, line:45:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_52:0x[a-z0-9]*]] <line:44:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_53:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-OverloadableAttr [[ADDR_54:0x[a-z0-9]*]] <line:8:37>
// C-NEXT: |-FunctionDecl [[ADDR_40]] <col:22, line:49:1> line:8:22 also_before[implementation={vendor(llvm)}] 'int (long)'
// C-NEXT: | |-ParmVarDecl [[ADDR_55:0x[a-z0-9]*]] <line:47:17, col:22> col:22 l 'long'
// C-NEXT: | |-CompoundStmt [[ADDR_56:0x[a-z0-9]*]] <col:25, line:49:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_57:0x[a-z0-9]*]] <line:48:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_58:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-OverloadableAttr [[ADDR_59:0x[a-z0-9]*]] <line:8:37>
// C-NEXT: `-FunctionDecl [[ADDR_60:0x[a-z0-9]*]] <line:53:1, line:56:1> line:53:5 main 'int ({{.*}})'
// C-NEXT:   `-CompoundStmt [[ADDR_61:0x[a-z0-9]*]] <col:16, line:56:1>
// C-NEXT:     `-ReturnStmt [[ADDR_62:0x[a-z0-9]*]] <line:55:3, col:96>
// C-NEXT:       `-BinaryOperator [[ADDR_63:0x[a-z0-9]*]] <col:10, col:96> 'int' '+'
// C-NEXT:         |-BinaryOperator [[ADDR_64:0x[a-z0-9]*]] <col:10, col:78> 'int' '+'
// C-NEXT:         | |-BinaryOperator [[ADDR_65:0x[a-z0-9]*]] <col:10, col:59> 'int' '+'
// C-NEXT:         | | |-BinaryOperator [[ADDR_66:0x[a-z0-9]*]] <col:10, col:39> 'int' '+'
// C-NEXT:         | | | |-PseudoObjectExpr [[ADDR_67:0x[a-z0-9]*]] <col:10, col:22> 'int'
// C-NEXT:         | | | | |-CallExpr [[ADDR_68:0x[a-z0-9]*]] <col:10, col:22> 'int'
// C-NEXT:         | | | | | `-ImplicitCastExpr [[ADDR_69:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | | | | |   `-DeclRefExpr [[ADDR_70:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// C-NEXT:         | | | | `-CallExpr [[ADDR_71:0x[a-z0-9]*]] <line:8:22, line:55:22> 'int'
// C-NEXT:         | | | |   `-ImplicitCastExpr [[ADDR_72:0x[a-z0-9]*]] <line:8:22> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | | | |     `-DeclRefExpr [[ADDR_6]] <col:22> 'int ({{.*}})' Function [[ADDR_7]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT:         | | | `-PseudoObjectExpr [[ADDR_73:0x[a-z0-9]*]] <line:55:26, col:39> 'int'
// C-NEXT:         | | |   |-CallExpr [[ADDR_74:0x[a-z0-9]*]] <col:26, col:39> 'int'
// C-NEXT:         | | |   | |-ImplicitCastExpr [[ADDR_75:0x[a-z0-9]*]] <col:26> 'int (*)(int)' <FunctionToPointerDecay>
// C-NEXT:         | | |   | | `-DeclRefExpr [[ADDR_76:0x[a-z0-9]*]] <col:26> 'int (int)' {{.*}}Function [[ADDR_8]] 'also_before' 'int (int)'
// C-NEXT:         | | |   | `-IntegerLiteral [[ADDR_77:0x[a-z0-9]*]] <col:38> 'int' 1
// C-NEXT:         | | |   `-CallExpr [[ADDR_78:0x[a-z0-9]*]] <line:8:22, line:55:39> 'int'
// C-NEXT:         | | |     |-ImplicitCastExpr [[ADDR_79:0x[a-z0-9]*]] <line:8:22> 'int (*)(int)' <FunctionToPointerDecay>
// C-NEXT:         | | |     | `-DeclRefExpr [[ADDR_15]] <col:22> 'int (int)' Function [[ADDR_16]] 'also_before[implementation={vendor(llvm)}]' 'int (int)'
// C-NEXT:         | | |     `-IntegerLiteral [[ADDR_77]] <line:55:38> 'int' 1
// C-NEXT:         | | `-CallExpr [[ADDR_80:0x[a-z0-9]*]] <col:43, col:59> 'int'
// C-NEXT:         | |   |-ImplicitCastExpr [[ADDR_81:0x[a-z0-9]*]] <col:43> 'int (*)(float)' <FunctionToPointerDecay>
// C-NEXT:         | |   | `-DeclRefExpr [[ADDR_82:0x[a-z0-9]*]] <col:43> 'int (float)' {{.*}}Function [[ADDR_17]] 'also_before' 'int (float)'
// C-NEXT:         | |   `-FloatingLiteral [[ADDR_83:0x[a-z0-9]*]] <col:55> 'float' 2.000000e+00
// C-NEXT:         | `-PseudoObjectExpr [[ADDR_84:0x[a-z0-9]*]] <col:63, col:78> 'int'
// C-NEXT:         |   |-CallExpr [[ADDR_85:0x[a-z0-9]*]] <col:63, col:78> 'int'
// C-NEXT:         |   | |-ImplicitCastExpr [[ADDR_86:0x[a-z0-9]*]] <col:63> 'int (*)(double)' <FunctionToPointerDecay>
// C-NEXT:         |   | | `-DeclRefExpr [[ADDR_87:0x[a-z0-9]*]] <col:63> 'int (double)' {{.*}}Function [[ADDR_23]] 'also_before' 'int (double)'
// C-NEXT:         |   | `-FloatingLiteral [[ADDR_88:0x[a-z0-9]*]] <col:75> 'double' 3.000000e+00
// C-NEXT:         |   `-CallExpr [[ADDR_89:0x[a-z0-9]*]] <line:8:22, line:55:78> 'int'
// C-NEXT:         |     |-ImplicitCastExpr [[ADDR_90:0x[a-z0-9]*]] <line:8:22> 'int (*)(double)' <FunctionToPointerDecay>
// C-NEXT:         |     | `-DeclRefExpr [[ADDR_30]] <col:22> 'int (double)' Function [[ADDR_31]] 'also_before[implementation={vendor(llvm)}]' 'int (double)'
// C-NEXT:         |     `-FloatingLiteral [[ADDR_88]] <line:55:75> 'double' 3.000000e+00
// C-NEXT:         `-PseudoObjectExpr [[ADDR_91:0x[a-z0-9]*]] <col:82, col:96> 'int'
// C-NEXT:           |-CallExpr [[ADDR_92:0x[a-z0-9]*]] <col:82, col:96> 'int'
// C-NEXT:           | |-ImplicitCastExpr [[ADDR_93:0x[a-z0-9]*]] <col:82> 'int (*)(long)' <FunctionToPointerDecay>
// C-NEXT:           | | `-DeclRefExpr [[ADDR_94:0x[a-z0-9]*]] <col:82> 'int (long)' {{.*}}Function [[ADDR_32]] 'also_before' 'int (long)'
// C-NEXT:           | `-IntegerLiteral [[ADDR_95:0x[a-z0-9]*]] <col:94> 'long' 4
// C-NEXT:           `-CallExpr [[ADDR_96:0x[a-z0-9]*]] <line:8:22, line:55:96> 'int'
// C-NEXT:             |-ImplicitCastExpr [[ADDR_97:0x[a-z0-9]*]] <line:8:22> 'int (*)(long)' <FunctionToPointerDecay>
// C-NEXT:             | `-DeclRefExpr [[ADDR_39]] <col:22> 'int (long)' Function [[ADDR_40]] 'also_before[implementation={vendor(llvm)}]' 'int (long)'
// C-NEXT:             `-IntegerLiteral [[ADDR_95]] <line:55:94> 'long' 4

// CXX:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:14:1> line:12:5 used also_before 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:14:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:13:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 1
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:34:1> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:16:1, line:18:1> line:16:5 used also_before 'int (int)'
// CXX-NEXT: | |-ParmVarDecl [[ADDR_8:0x[a-z0-9]*]] <col:17, col:21> col:21 i 'int'
// CXX-NEXT: | |-CompoundStmt [[ADDR_9:0x[a-z0-9]*]] <col:24, line:18:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_10:0x[a-z0-9]*]] <line:17:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_11:0x[a-z0-9]*]] <col:10> 'int' 2
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_12:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_13:0x[a-z0-9]*]] <line:38:1> 'int (int)' Function [[ADDR_14:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int (int)'
// CXX-NEXT: |-FunctionDecl [[ADDR_15:0x[a-z0-9]*]] <line:20:1, line:22:1> line:20:5 used also_before 'int (float)'
// CXX-NEXT: | |-ParmVarDecl [[ADDR_16:0x[a-z0-9]*]] <col:17, col:23> col:23 f 'float'
// CXX-NEXT: | `-CompoundStmt [[ADDR_17:0x[a-z0-9]*]] <col:26, line:22:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_18:0x[a-z0-9]*]] <line:21:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_19:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_20:0x[a-z0-9]*]] <line:24:1, line:26:1> line:24:5 used also_before 'int (double)'
// CXX-NEXT: | |-ParmVarDecl [[ADDR_21:0x[a-z0-9]*]] <col:17, col:24> col:24 d 'double'
// CXX-NEXT: | |-CompoundStmt [[ADDR_22:0x[a-z0-9]*]] <col:27, line:26:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_23:0x[a-z0-9]*]] <line:25:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_24:0x[a-z0-9]*]] <col:10> 'int' 3
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_25:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_26:0x[a-z0-9]*]] <line:43:1> 'int (double)' Function [[ADDR_27:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int (double)'
// CXX-NEXT: |-FunctionDecl [[ADDR_28:0x[a-z0-9]*]] <line:28:1, line:30:1> line:28:5 used also_before 'int (long)'
// CXX-NEXT: | |-ParmVarDecl [[ADDR_29:0x[a-z0-9]*]] <col:17, col:22> col:22 l 'long'
// CXX-NEXT: | |-CompoundStmt [[ADDR_30:0x[a-z0-9]*]] <col:25, line:30:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_31:0x[a-z0-9]*]] <line:29:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_32:0x[a-z0-9]*]] <col:10> 'int' 4
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_33:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_34:0x[a-z0-9]*]] <line:47:1> 'int (long)' Function [[ADDR_35:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int (long)'
// CXX-NEXT: |-FunctionDecl [[ADDR_6]] <line:34:1, line:36:1> line:34:1 also_before[implementation={vendor(llvm)}] 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_36:0x[a-z0-9]*]] <col:23, line:36:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_37:0x[a-z0-9]*]] <line:35:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_38:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_14]] <line:38:1, line:40:1> line:38:1 also_before[implementation={vendor(llvm)}] 'int (int)'
// CXX-NEXT: | |-ParmVarDecl [[ADDR_39:0x[a-z0-9]*]] <col:17, col:21> col:21 i 'int'
// CXX-NEXT: | `-CompoundStmt [[ADDR_40:0x[a-z0-9]*]] <col:24, line:40:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_41:0x[a-z0-9]*]] <line:39:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_42:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_27]] <line:43:1, line:45:1> line:43:1 also_before[implementation={vendor(llvm)}] 'int (double)'
// CXX-NEXT: | |-ParmVarDecl [[ADDR_43:0x[a-z0-9]*]] <col:17, col:24> col:24 d 'double'
// CXX-NEXT: | `-CompoundStmt [[ADDR_44:0x[a-z0-9]*]] <col:27, line:45:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_45:0x[a-z0-9]*]] <line:44:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_46:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_35]] <line:47:1, line:49:1> line:47:1 also_before[implementation={vendor(llvm)}] 'int (long)'
// CXX-NEXT: | |-ParmVarDecl [[ADDR_47:0x[a-z0-9]*]] <col:17, col:22> col:22 l 'long'
// CXX-NEXT: | `-CompoundStmt [[ADDR_48:0x[a-z0-9]*]] <col:25, line:49:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_49:0x[a-z0-9]*]] <line:48:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_50:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: `-FunctionDecl [[ADDR_51:0x[a-z0-9]*]] <line:53:1, line:56:1> line:53:5 main 'int ({{.*}})'
// CXX-NEXT:   `-CompoundStmt [[ADDR_52:0x[a-z0-9]*]] <col:16, line:56:1>
// CXX-NEXT:     `-ReturnStmt [[ADDR_53:0x[a-z0-9]*]] <line:55:3, col:96>
// CXX-NEXT:       `-BinaryOperator [[ADDR_54:0x[a-z0-9]*]] <col:10, col:96> 'int' '+'
// CXX-NEXT:         |-BinaryOperator [[ADDR_55:0x[a-z0-9]*]] <col:10, col:78> 'int' '+'
// CXX-NEXT:         | |-BinaryOperator [[ADDR_56:0x[a-z0-9]*]] <col:10, col:59> 'int' '+'
// CXX-NEXT:         | | |-BinaryOperator [[ADDR_57:0x[a-z0-9]*]] <col:10, col:39> 'int' '+'
// CXX-NEXT:         | | | |-PseudoObjectExpr [[ADDR_58:0x[a-z0-9]*]] <col:10, col:22> 'int'
// CXX-NEXT:         | | | | |-CallExpr [[ADDR_59:0x[a-z0-9]*]] <col:10, col:22> 'int'
// CXX-NEXT:         | | | | | `-ImplicitCastExpr [[ADDR_60:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | | | | |   `-DeclRefExpr [[ADDR_61:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CXX-NEXT:         | | | | `-CallExpr [[ADDR_62:0x[a-z0-9]*]] <line:34:1, line:55:22> 'int'
// CXX-NEXT:         | | | |   `-ImplicitCastExpr [[ADDR_63:0x[a-z0-9]*]] <line:34:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | | | |     `-DeclRefExpr [[ADDR_5]] <col:1> 'int ({{.*}})' Function [[ADDR_6]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT:         | | | `-PseudoObjectExpr [[ADDR_64:0x[a-z0-9]*]] <line:55:26, col:39> 'int'
// CXX-NEXT:         | | |   |-CallExpr [[ADDR_65:0x[a-z0-9]*]] <col:26, col:39> 'int'
// CXX-NEXT:         | | |   | |-ImplicitCastExpr [[ADDR_66:0x[a-z0-9]*]] <col:26> 'int (*)(int)' <FunctionToPointerDecay>
// CXX-NEXT:         | | |   | | `-DeclRefExpr [[ADDR_67:0x[a-z0-9]*]] <col:26> 'int (int)' {{.*}}Function [[ADDR_7]] 'also_before' 'int (int)'
// CXX-NEXT:         | | |   | `-IntegerLiteral [[ADDR_68:0x[a-z0-9]*]] <col:38> 'int' 1
// CXX-NEXT:         | | |   `-CallExpr [[ADDR_69:0x[a-z0-9]*]] <line:38:1, line:55:39> 'int'
// CXX-NEXT:         | | |     |-ImplicitCastExpr [[ADDR_70:0x[a-z0-9]*]] <line:38:1> 'int (*)(int)' <FunctionToPointerDecay>
// CXX-NEXT:         | | |     | `-DeclRefExpr [[ADDR_13]] <col:1> 'int (int)' Function [[ADDR_14]] 'also_before[implementation={vendor(llvm)}]' 'int (int)'
// CXX-NEXT:         | | |     `-IntegerLiteral [[ADDR_68]] <line:55:38> 'int' 1
// CXX-NEXT:         | | `-CallExpr [[ADDR_71:0x[a-z0-9]*]] <col:43, col:59> 'int'
// CXX-NEXT:         | |   |-ImplicitCastExpr [[ADDR_72:0x[a-z0-9]*]] <col:43> 'int (*)(float)' <FunctionToPointerDecay>
// CXX-NEXT:         | |   | `-DeclRefExpr [[ADDR_73:0x[a-z0-9]*]] <col:43> 'int (float)' {{.*}}Function [[ADDR_15]] 'also_before' 'int (float)'
// CXX-NEXT:         | |   `-FloatingLiteral [[ADDR_74:0x[a-z0-9]*]] <col:55> 'float' 2.000000e+00
// CXX-NEXT:         | `-PseudoObjectExpr [[ADDR_75:0x[a-z0-9]*]] <col:63, col:78> 'int'
// CXX-NEXT:         |   |-CallExpr [[ADDR_76:0x[a-z0-9]*]] <col:63, col:78> 'int'
// CXX-NEXT:         |   | |-ImplicitCastExpr [[ADDR_77:0x[a-z0-9]*]] <col:63> 'int (*)(double)' <FunctionToPointerDecay>
// CXX-NEXT:         |   | | `-DeclRefExpr [[ADDR_78:0x[a-z0-9]*]] <col:63> 'int (double)' {{.*}}Function [[ADDR_20]] 'also_before' 'int (double)'
// CXX-NEXT:         |   | `-FloatingLiteral [[ADDR_79:0x[a-z0-9]*]] <col:75> 'double' 3.000000e+00
// CXX-NEXT:         |   `-CallExpr [[ADDR_80:0x[a-z0-9]*]] <line:43:1, line:55:78> 'int'
// CXX-NEXT:         |     |-ImplicitCastExpr [[ADDR_81:0x[a-z0-9]*]] <line:43:1> 'int (*)(double)' <FunctionToPointerDecay>
// CXX-NEXT:         |     | `-DeclRefExpr [[ADDR_26]] <col:1> 'int (double)' Function [[ADDR_27]] 'also_before[implementation={vendor(llvm)}]' 'int (double)'
// CXX-NEXT:         |     `-FloatingLiteral [[ADDR_79]] <line:55:75> 'double' 3.000000e+00
// CXX-NEXT:         `-PseudoObjectExpr [[ADDR_82:0x[a-z0-9]*]] <col:82, col:96> 'int'
// CXX-NEXT:           |-CallExpr [[ADDR_83:0x[a-z0-9]*]] <col:82, col:96> 'int'
// CXX-NEXT:           | |-ImplicitCastExpr [[ADDR_84:0x[a-z0-9]*]] <col:82> 'int (*)(long)' <FunctionToPointerDecay>
// CXX-NEXT:           | | `-DeclRefExpr [[ADDR_85:0x[a-z0-9]*]] <col:82> 'int (long)' {{.*}}Function [[ADDR_28]] 'also_before' 'int (long)'
// CXX-NEXT:           | `-IntegerLiteral [[ADDR_86:0x[a-z0-9]*]] <col:94> 'long' 4
// CXX-NEXT:           `-CallExpr [[ADDR_87:0x[a-z0-9]*]] <line:47:1, line:55:96> 'int'
// CXX-NEXT:             |-ImplicitCastExpr [[ADDR_88:0x[a-z0-9]*]] <line:47:1> 'int (*)(long)' <FunctionToPointerDecay>
// CXX-NEXT:             | `-DeclRefExpr [[ADDR_34]] <col:1> 'int (long)' Function [[ADDR_35]] 'also_before[implementation={vendor(llvm)}]' 'int (long)'
// CXX-NEXT:             `-IntegerLiteral [[ADDR_86]] <line:55:94> 'long' 4
