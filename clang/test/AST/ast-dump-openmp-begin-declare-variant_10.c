// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s --check-prefix=C
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s --check-prefix=CXX
// expected-no-diagnostics

#ifdef __cplusplus
#define CONST constexpr
#else
#define CONST __attribute__((const))
#endif

int also_before1(void) {
  return 1;
}
int also_before2(void) {
  return 2;
}
int also_before3(void) {
  return 3;
}
int also_before4(void) {
  return 4;
}

#pragma omp begin declare variant match(implementation = {vendor(llvm)})
CONST int also_before1(void) {
  return 0;
}
static int also_before2(void) {
  return 0;
}
__attribute__((nothrow)) int also_before3(void) {
  return 0;
}
static CONST __attribute__((nothrow, always_inline)) __inline__ int also_before4(void) {
  return 0;
}
#pragma omp end declare variant


int main() {
  // Should return 0.
  return also_before1() + also_before2() + also_before3() + also_before4();
}

// Make sure:
//  - we see the specialization in the AST
//  - we pick the right callees

// C:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:13:1> line:11:5 used also_before1 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:24, line:13:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:12:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 1
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:8:15> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:14:1, line:16:1> line:14:5 used also_before2 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_8:0x[a-z0-9]*]] <col:24, line:16:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_9:0x[a-z0-9]*]] <line:15:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_10:0x[a-z0-9]*]] <col:10> 'int' 2
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_11:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_12:0x[a-z0-9]*]] <line:28:1> 'int ({{.*}})' Function [[ADDR_13:0x[a-z0-9]*]] 'also_before2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_14:0x[a-z0-9]*]] <line:17:1, line:19:1> line:17:5 used also_before3 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_15:0x[a-z0-9]*]] <col:24, line:19:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_16:0x[a-z0-9]*]] <line:18:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_17:0x[a-z0-9]*]] <col:10> 'int' 3
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_18:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_19:0x[a-z0-9]*]] <line:31:1> 'int ({{.*}})' Function [[ADDR_20:0x[a-z0-9]*]] 'also_before3[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_21:0x[a-z0-9]*]] <line:20:1, line:22:1> line:20:5 used also_before4 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_22:0x[a-z0-9]*]] <col:24, line:22:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_23:0x[a-z0-9]*]] <line:21:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_24:0x[a-z0-9]*]] <col:10> 'int' 4
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_25:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_26:0x[a-z0-9]*]] <line:34:1> 'int ({{.*}})' Function [[ADDR_27:0x[a-z0-9]*]] 'also_before4[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_6]] <line:8:15, line:27:1> line:8:15 also_before1[implementation={vendor(llvm)}] 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_28:0x[a-z0-9]*]] <line:25:30, line:27:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_29:0x[a-z0-9]*]] <line:26:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_30:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-ConstAttr [[ADDR_31:0x[a-z0-9]*]] <line:8:30>
// C-NEXT: |-FunctionDecl [[ADDR_13]] <line:28:1, line:30:1> line:28:1 also_before2[implementation={vendor(llvm)}] 'int ({{.*}})' static
// C-NEXT: | `-CompoundStmt [[ADDR_32:0x[a-z0-9]*]] <col:31, line:30:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_33:0x[a-z0-9]*]] <line:29:3, col:10>
// C-NEXT: |     `-IntegerLiteral [[ADDR_34:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: |-FunctionDecl [[ADDR_20]] <line:31:1, line:33:1> line:31:1 also_before3[implementation={vendor(llvm)}] 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_35:0x[a-z0-9]*]] <col:49, line:33:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_36:0x[a-z0-9]*]] <line:32:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_37:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-NoThrowAttr [[ADDR_38:0x[a-z0-9]*]] <line:31:16>
// C-NEXT: |-FunctionDecl [[ADDR_27]] <line:34:1, line:36:1> line:34:1 also_before4[implementation={vendor(llvm)}] 'int ({{.*}})' static inline
// C-NEXT: | |-CompoundStmt [[ADDR_39:0x[a-z0-9]*]] <col:88, line:36:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_40:0x[a-z0-9]*]] <line:35:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_41:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | |-ConstAttr [[ADDR_42:0x[a-z0-9]*]] <line:8:30>
// C-NEXT: | |-NoThrowAttr [[ADDR_43:0x[a-z0-9]*]] <line:34:29>
// C-NEXT: | `-AlwaysInlineAttr [[ADDR_44:0x[a-z0-9]*]] <col:38> always_inline
// C-NEXT: `-FunctionDecl [[ADDR_45:0x[a-z0-9]*]] <line:40:1, line:43:1> line:40:5 main 'int ({{.*}})'
// C-NEXT:   `-CompoundStmt [[ADDR_46:0x[a-z0-9]*]] <col:12, line:43:1>
// C-NEXT:     `-ReturnStmt [[ADDR_47:0x[a-z0-9]*]] <line:42:3, col:74>
// C-NEXT:       `-BinaryOperator [[ADDR_48:0x[a-z0-9]*]] <col:10, col:74> 'int' '+'
// C-NEXT:         |-BinaryOperator [[ADDR_49:0x[a-z0-9]*]] <col:10, col:57> 'int' '+'
// C-NEXT:         | |-BinaryOperator [[ADDR_50:0x[a-z0-9]*]] <col:10, col:40> 'int' '+'
// C-NEXT:         | | |-PseudoObjectExpr [[ADDR_51:0x[a-z0-9]*]] <col:10, col:23> 'int'
// C-NEXT:         | | | |-CallExpr [[ADDR_52:0x[a-z0-9]*]] <col:10, col:23> 'int'
// C-NEXT:         | | | | `-ImplicitCastExpr [[ADDR_53:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | | | |   `-DeclRefExpr [[ADDR_54:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' Function [[ADDR_0]] 'also_before1' 'int ({{.*}})'
// C-NEXT:         | | | `-CallExpr [[ADDR_55:0x[a-z0-9]*]] <line:8:15, line:42:23> 'int'
// C-NEXT:         | | |   `-ImplicitCastExpr [[ADDR_56:0x[a-z0-9]*]] <line:8:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | | |     `-DeclRefExpr [[ADDR_5]] <col:15> 'int ({{.*}})' Function [[ADDR_6]] 'also_before1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT:         | | `-PseudoObjectExpr [[ADDR_57:0x[a-z0-9]*]] <line:42:27, col:40> 'int'
// C-NEXT:         | |   |-CallExpr [[ADDR_58:0x[a-z0-9]*]] <col:27, col:40> 'int'
// C-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_59:0x[a-z0-9]*]] <col:27> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | |   |   `-DeclRefExpr [[ADDR_60:0x[a-z0-9]*]] <col:27> 'int ({{.*}})' Function [[ADDR_7]] 'also_before2' 'int ({{.*}})'
// C-NEXT:         | |   `-CallExpr [[ADDR_61:0x[a-z0-9]*]] <line:28:1, line:42:40> 'int'
// C-NEXT:         | |     `-ImplicitCastExpr [[ADDR_62:0x[a-z0-9]*]] <line:28:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | |       `-DeclRefExpr [[ADDR_12]] <col:1> 'int ({{.*}})' Function [[ADDR_13]] 'also_before2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT:         | `-PseudoObjectExpr [[ADDR_63:0x[a-z0-9]*]] <line:42:44, col:57> 'int'
// C-NEXT:         |   |-CallExpr [[ADDR_64:0x[a-z0-9]*]] <col:44, col:57> 'int'
// C-NEXT:         |   | `-ImplicitCastExpr [[ADDR_65:0x[a-z0-9]*]] <col:44> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         |   |   `-DeclRefExpr [[ADDR_66:0x[a-z0-9]*]] <col:44> 'int ({{.*}})' Function [[ADDR_14]] 'also_before3' 'int ({{.*}})'
// C-NEXT:         |   `-CallExpr [[ADDR_67:0x[a-z0-9]*]] <line:31:1, line:42:57> 'int'
// C-NEXT:         |     `-ImplicitCastExpr [[ADDR_68:0x[a-z0-9]*]] <line:31:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         |       `-DeclRefExpr [[ADDR_19]] <col:1> 'int ({{.*}})' Function [[ADDR_20]] 'also_before3[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT:         `-PseudoObjectExpr [[ADDR_69:0x[a-z0-9]*]] <line:42:61, col:74> 'int'
// C-NEXT:           |-CallExpr [[ADDR_70:0x[a-z0-9]*]] <col:61, col:74> 'int'
// C-NEXT:           | `-ImplicitCastExpr [[ADDR_71:0x[a-z0-9]*]] <col:61> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:           |   `-DeclRefExpr [[ADDR_72:0x[a-z0-9]*]] <col:61> 'int ({{.*}})' Function [[ADDR_21]] 'also_before4' 'int ({{.*}})'
// C-NEXT:           `-CallExpr [[ADDR_73:0x[a-z0-9]*]] <line:34:1, line:42:74> 'int'
// C-NEXT:             `-ImplicitCastExpr [[ADDR_74:0x[a-z0-9]*]] <line:34:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:               `-DeclRefExpr [[ADDR_26]] <col:1> 'int ({{.*}})' Function [[ADDR_27]] 'also_before4[implementation={vendor(llvm)}]' 'int ({{.*}})'

// CXX:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:13:1> line:11:5 used also_before1 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:24, line:13:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:12:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 1
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:6:15> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:14:1, line:16:1> line:14:5 used also_before2 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_8:0x[a-z0-9]*]] <col:24, line:16:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_9:0x[a-z0-9]*]] <line:15:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_10:0x[a-z0-9]*]] <col:10> 'int' 2
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_11:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_12:0x[a-z0-9]*]] <line:28:1> 'int ({{.*}})' Function [[ADDR_13:0x[a-z0-9]*]] 'also_before2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_14:0x[a-z0-9]*]] <line:17:1, line:19:1> line:17:5 used also_before3 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_15:0x[a-z0-9]*]] <col:24, line:19:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_16:0x[a-z0-9]*]] <line:18:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_17:0x[a-z0-9]*]] <col:10> 'int' 3
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_18:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_19:0x[a-z0-9]*]] <line:31:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_20:0x[a-z0-9]*]] 'also_before3[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT: |-FunctionDecl [[ADDR_21:0x[a-z0-9]*]] <line:20:1, line:22:1> line:20:5 used also_before4 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_22:0x[a-z0-9]*]] <col:24, line:22:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_23:0x[a-z0-9]*]] <line:21:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_24:0x[a-z0-9]*]] <col:10> 'int' 4
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_25:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_26:0x[a-z0-9]*]] <line:34:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_27:0x[a-z0-9]*]] 'also_before4[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT: |-FunctionDecl [[ADDR_6]] <line:6:15, line:27:1> line:6:15 constexpr also_before1[implementation={vendor(llvm)}] 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_28:0x[a-z0-9]*]] <line:25:30, line:27:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_29:0x[a-z0-9]*]] <line:26:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_30:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_13]] <line:28:1, line:30:1> line:28:1 also_before2[implementation={vendor(llvm)}] 'int ({{.*}})' static
// CXX-NEXT: | `-CompoundStmt [[ADDR_31:0x[a-z0-9]*]] <col:31, line:30:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_32:0x[a-z0-9]*]] <line:29:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_33:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_20]] <line:31:1, line:33:1> line:31:1 also_before3[implementation={vendor(llvm)}] 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT: | `-CompoundStmt [[ADDR_34:0x[a-z0-9]*]] <col:49, line:33:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_35:0x[a-z0-9]*]] <line:32:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_36:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_27]] <line:34:1, line:36:1> line:34:1 constexpr also_before4[implementation={vendor(llvm)}] 'int ({{.*}}) __attribute__((nothrow))' static inline
// CXX-NEXT: | |-CompoundStmt [[ADDR_37:0x[a-z0-9]*]] <col:88, line:36:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_38:0x[a-z0-9]*]] <line:35:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_39:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: | `-AlwaysInlineAttr [[ADDR_40:0x[a-z0-9]*]] <line:34:38> always_inline
// CXX-NEXT: `-FunctionDecl [[ADDR_41:0x[a-z0-9]*]] <line:40:1, line:43:1> line:40:5 main 'int ({{.*}})'
// CXX-NEXT:   `-CompoundStmt [[ADDR_42:0x[a-z0-9]*]] <col:12, line:43:1>
// CXX-NEXT:     `-ReturnStmt [[ADDR_43:0x[a-z0-9]*]] <line:42:3, col:74>
// CXX-NEXT:       `-BinaryOperator [[ADDR_44:0x[a-z0-9]*]] <col:10, col:74> 'int' '+'
// CXX-NEXT:         |-BinaryOperator [[ADDR_45:0x[a-z0-9]*]] <col:10, col:57> 'int' '+'
// CXX-NEXT:         | |-BinaryOperator [[ADDR_46:0x[a-z0-9]*]] <col:10, col:40> 'int' '+'
// CXX-NEXT:         | | |-PseudoObjectExpr [[ADDR_47:0x[a-z0-9]*]] <col:10, col:23> 'int'
// CXX-NEXT:         | | | |-CallExpr [[ADDR_48:0x[a-z0-9]*]] <col:10, col:23> 'int'
// CXX-NEXT:         | | | | `-ImplicitCastExpr [[ADDR_49:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | | | |   `-DeclRefExpr [[ADDR_50:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before1' 'int ({{.*}})'
// CXX-NEXT:         | | | `-CallExpr [[ADDR_51:0x[a-z0-9]*]] <line:6:15, line:42:23> 'int'
// CXX-NEXT:         | | |   `-ImplicitCastExpr [[ADDR_52:0x[a-z0-9]*]] <line:6:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | | |     `-DeclRefExpr [[ADDR_5]] <col:15> 'int ({{.*}})' Function [[ADDR_6]] 'also_before1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT:         | | `-PseudoObjectExpr [[ADDR_53:0x[a-z0-9]*]] <line:42:27, col:40> 'int'
// CXX-NEXT:         | |   |-CallExpr [[ADDR_54:0x[a-z0-9]*]] <col:27, col:40> 'int'
// CXX-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_55:0x[a-z0-9]*]] <col:27> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | |   |   `-DeclRefExpr [[ADDR_56:0x[a-z0-9]*]] <col:27> 'int ({{.*}})' {{.*}}Function [[ADDR_7]] 'also_before2' 'int ({{.*}})'
// CXX-NEXT:         | |   `-CallExpr [[ADDR_57:0x[a-z0-9]*]] <line:28:1, line:42:40> 'int'
// CXX-NEXT:         | |     `-ImplicitCastExpr [[ADDR_58:0x[a-z0-9]*]] <line:28:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | |       `-DeclRefExpr [[ADDR_12]] <col:1> 'int ({{.*}})' Function [[ADDR_13]] 'also_before2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT:         | `-PseudoObjectExpr [[ADDR_59:0x[a-z0-9]*]] <line:42:44, col:57> 'int'
// CXX-NEXT:         |   |-CallExpr [[ADDR_60:0x[a-z0-9]*]] <col:44, col:57> 'int'
// CXX-NEXT:         |   | `-ImplicitCastExpr [[ADDR_61:0x[a-z0-9]*]] <col:44> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         |   |   `-DeclRefExpr [[ADDR_62:0x[a-z0-9]*]] <col:44> 'int ({{.*}})' {{.*}}Function [[ADDR_14]] 'also_before3' 'int ({{.*}})'
// CXX-NEXT:         |   `-CallExpr [[ADDR_63:0x[a-z0-9]*]] <line:31:1, line:42:57> 'int'
// CXX-NEXT:         |     `-ImplicitCastExpr [[ADDR_64:0x[a-z0-9]*]] <line:31:1> 'int (*)({{.*}}) __attribute__((nothrow))' <FunctionToPointerDecay>
// CXX-NEXT:         |       `-DeclRefExpr [[ADDR_19]] <col:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_20]] 'also_before3[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT:         `-PseudoObjectExpr [[ADDR_65:0x[a-z0-9]*]] <line:42:61, col:74> 'int'
// CXX-NEXT:           |-CallExpr [[ADDR_66:0x[a-z0-9]*]] <col:61, col:74> 'int'
// CXX-NEXT:           | `-ImplicitCastExpr [[ADDR_67:0x[a-z0-9]*]] <col:61> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:           |   `-DeclRefExpr [[ADDR_68:0x[a-z0-9]*]] <col:61> 'int ({{.*}})' {{.*}}Function [[ADDR_21]] 'also_before4' 'int ({{.*}})'
// CXX-NEXT:           `-CallExpr [[ADDR_69:0x[a-z0-9]*]] <line:34:1, line:42:74> 'int'
// CXX-NEXT:             `-ImplicitCastExpr [[ADDR_70:0x[a-z0-9]*]] <line:34:1> 'int (*)({{.*}}) __attribute__((nothrow))' <FunctionToPointerDecay>
// CXX-NEXT:               `-DeclRefExpr [[ADDR_26]] <col:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_27]] 'also_before4[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
