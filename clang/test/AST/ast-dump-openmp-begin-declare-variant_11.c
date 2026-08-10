// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=c_mode   -ast-dump %s       | FileCheck %s --check-prefix=C
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=cxx_mode -ast-dump %s -x c++ %std_cxx11-14 | FileCheck %s --check-prefix=CXX
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=cxx_mode,cxx17 -ast-dump %s -x c++ %std_cxx17- | FileCheck %s --check-prefix=CXX
// c_mode-no-diagnostics

#ifdef __cplusplus
#define CONST constexpr
#else
#define CONST __attribute__((const))
#endif

#pragma omp begin declare variant match(implementation = {vendor(llvm)})
CONST int also_after1(void) { // cxx_mode-note {{previous declaration is here}}
  return 0;
}
static int also_after2(void) {
  return 0;
}
__attribute__((nothrow)) int also_after3(void) { // cxx17-note {{previous declaration is here}}
  return 0;
}
static CONST __attribute__((nothrow, always_inline)) __inline__ int also_after4(void) { // cxx_mode-note {{previous declaration is here}} cxx17-note {{previous declaration is here}}
  return 0;
}
#pragma omp end declare variant

int also_after1(void) { // cxx_mode-error {{non-constexpr declaration of 'also_after1' follows constexpr declaration}}
  return 1;
}
int also_after2(void) {
  return 2;
}
int also_after3(void) { // cxx17-warning {{'also_after3' is missing exception specification '__attribute__((nothrow))'}}
  return 3;
}
int also_after4(void) { // cxx_mode-error {{non-constexpr declaration of 'also_after4' follows constexpr declaration}} cxx17-warning {{'also_after4' is missing exception specification '__attribute__((nothrow))'}}
  return 4;
}


int main(void) {
  // Should return 0.
  return also_after1() + also_after2() + also_after3() + also_after4();
}

// Make sure:
//  - we see the specialization in the AST
//  - we pick the right callees

// C:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:13:27> col:11 implicit used also_after1 'int ({{.*}})'
// C-NEXT: | |-ConstAttr [[ADDR_1:0x[a-z0-9]*]] <line:9:30>
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_2:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_3:0x[a-z0-9]*]] <col:15> 'int ({{.*}})' Function [[ADDR_4:0x[a-z0-9]*]] 'also_after1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_4]] <col:15, line:15:1> line:9:15 also_after1[implementation={vendor(llvm)}] 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_5:0x[a-z0-9]*]] <line:13:29, line:15:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_6:0x[a-z0-9]*]] <line:14:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_7:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-ConstAttr [[ADDR_8:0x[a-z0-9]*]] <line:9:30>
// C-NEXT: |-FunctionDecl [[ADDR_9:0x[a-z0-9]*]] <line:16:1, col:28> col:12 implicit used also_after2 'int ({{.*}})' static
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_10:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_11:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_12:0x[a-z0-9]*]] 'also_after2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_12]] <col:1, line:18:1> line:16:1 also_after2[implementation={vendor(llvm)}] 'int ({{.*}})' static
// C-NEXT: | `-CompoundStmt [[ADDR_13:0x[a-z0-9]*]] <col:30, line:18:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_14:0x[a-z0-9]*]] <line:17:3, col:10>
// C-NEXT: |     `-IntegerLiteral [[ADDR_15:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: |-FunctionDecl [[ADDR_16:0x[a-z0-9]*]] <line:19:1, col:46> col:30 implicit used also_after3 'int ({{.*}})'
// C-NEXT: | |-NoThrowAttr [[ADDR_17:0x[a-z0-9]*]] <col:16>
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_18:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_19:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_20:0x[a-z0-9]*]] 'also_after3[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_20]] <col:1, line:21:1> line:19:1 also_after3[implementation={vendor(llvm)}] 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_21:0x[a-z0-9]*]] <col:48, line:21:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_22:0x[a-z0-9]*]] <line:20:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_23:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-NoThrowAttr [[ADDR_24:0x[a-z0-9]*]] <line:19:16>
// C-NEXT: |-FunctionDecl [[ADDR_25:0x[a-z0-9]*]] <line:22:1, col:85> col:69 implicit used also_after4 'int ({{.*}})' static inline
// C-NEXT: | |-ConstAttr [[ADDR_26:0x[a-z0-9]*]] <line:9:30>
// C-NEXT: | |-NoThrowAttr [[ADDR_27:0x[a-z0-9]*]] <line:22:29>
// C-NEXT: | |-AlwaysInlineAttr [[ADDR_28:0x[a-z0-9]*]] <col:38> always_inline
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_29:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_30:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_31:0x[a-z0-9]*]] 'also_after4[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_31]] <col:1, line:24:1> line:22:1 also_after4[implementation={vendor(llvm)}] 'int ({{.*}})' static inline
// C-NEXT: | |-CompoundStmt [[ADDR_32:0x[a-z0-9]*]] <col:87, line:24:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_33:0x[a-z0-9]*]] <line:23:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_34:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | |-ConstAttr [[ADDR_35:0x[a-z0-9]*]] <line:9:30>
// C-NEXT: | |-NoThrowAttr [[ADDR_36:0x[a-z0-9]*]] <line:22:29>
// C-NEXT: | `-AlwaysInlineAttr [[ADDR_37:0x[a-z0-9]*]] <col:38> always_inline
// C-NEXT: |-FunctionDecl [[ADDR_38:0x[a-z0-9]*]] prev [[ADDR_0]] <line:27:1, line:29:1> line:27:5 used also_after1 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_39:0x[a-z0-9]*]] <col:23, line:29:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_40:0x[a-z0-9]*]] <line:28:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_41:0x[a-z0-9]*]] <col:10> 'int' 1
// C-NEXT: | |-ConstAttr [[ADDR_42:0x[a-z0-9]*]] <line:9:30> Inherited
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_43:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_3]] <col:15> 'int ({{.*}})' Function [[ADDR_4]] 'also_after1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_44:0x[a-z0-9]*]] prev [[ADDR_9]] <line:30:1, line:32:1> line:30:5 used also_after2 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_45:0x[a-z0-9]*]] <col:23, line:32:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_46:0x[a-z0-9]*]] <line:31:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_47:0x[a-z0-9]*]] <col:10> 'int' 2
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_48:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_11]] <line:16:1> 'int ({{.*}})' Function [[ADDR_12]] 'also_after2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_49:0x[a-z0-9]*]] prev [[ADDR_16]] <line:33:1, line:35:1> line:33:5 used also_after3 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_50:0x[a-z0-9]*]] <col:23, line:35:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_51:0x[a-z0-9]*]] <line:34:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_52:0x[a-z0-9]*]] <col:10> 'int' 3
// C-NEXT: | |-NoThrowAttr [[ADDR_53:0x[a-z0-9]*]] <line:19:16> Inherited
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_54:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_19]] <col:1> 'int ({{.*}})' Function [[ADDR_20]] 'also_after3[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_55:0x[a-z0-9]*]] prev [[ADDR_25]] <line:36:1, line:38:1> line:36:5 used also_after4 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_56:0x[a-z0-9]*]] <col:23, line:38:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_57:0x[a-z0-9]*]] <line:37:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_58:0x[a-z0-9]*]] <col:10> 'int' 4
// C-NEXT: | |-ConstAttr [[ADDR_59:0x[a-z0-9]*]] <line:9:30> Inherited
// C-NEXT: | |-NoThrowAttr [[ADDR_60:0x[a-z0-9]*]] <line:22:29> Inherited
// C-NEXT: | |-AlwaysInlineAttr [[ADDR_61:0x[a-z0-9]*]] <col:38> Inherited always_inline
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_62:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_30]] <col:1> 'int ({{.*}})' Function [[ADDR_31]] 'also_after4[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: `-FunctionDecl [[ADDR_63:0x[a-z0-9]*]] <line:41:1, line:44:1> line:41:5 main 'int ({{.*}})'
// C-NEXT:   `-CompoundStmt [[ADDR_64:0x[a-z0-9]*]] <col:16, line:44:1>
// C-NEXT:     `-ReturnStmt [[ADDR_65:0x[a-z0-9]*]] <line:43:3, col:70>
// C-NEXT:       `-BinaryOperator [[ADDR_66:0x[a-z0-9]*]] <col:10, col:70> 'int' '+'
// C-NEXT:         |-BinaryOperator [[ADDR_67:0x[a-z0-9]*]] <col:10, col:54> 'int' '+'
// C-NEXT:         | |-BinaryOperator [[ADDR_68:0x[a-z0-9]*]] <col:10, col:38> 'int' '+'
// C-NEXT:         | | |-PseudoObjectExpr [[ADDR_69:0x[a-z0-9]*]] <col:10, col:22> 'int'
// C-NEXT:         | | | |-CallExpr [[ADDR_70:0x[a-z0-9]*]] <col:10, col:22> 'int'
// C-NEXT:         | | | | `-ImplicitCastExpr [[ADDR_71:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | | | |   `-DeclRefExpr [[ADDR_72:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' Function [[ADDR_38]] 'also_after1' 'int ({{.*}})'
// C-NEXT:         | | | `-CallExpr [[ADDR_73:0x[a-z0-9]*]] <line:9:15, line:43:22> 'int'
// C-NEXT:         | | |   `-ImplicitCastExpr [[ADDR_74:0x[a-z0-9]*]] <line:9:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | | |     `-DeclRefExpr [[ADDR_3]] <col:15> 'int ({{.*}})' Function [[ADDR_4]] 'also_after1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT:         | | `-PseudoObjectExpr [[ADDR_75:0x[a-z0-9]*]] <line:43:26, col:38> 'int'
// C-NEXT:         | |   |-CallExpr [[ADDR_76:0x[a-z0-9]*]] <col:26, col:38> 'int'
// C-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_77:0x[a-z0-9]*]] <col:26> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | |   |   `-DeclRefExpr [[ADDR_78:0x[a-z0-9]*]] <col:26> 'int ({{.*}})' Function [[ADDR_44]] 'also_after2' 'int ({{.*}})'
// C-NEXT:         | |   `-CallExpr [[ADDR_79:0x[a-z0-9]*]] <line:16:1, line:43:38> 'int'
// C-NEXT:         | |     `-ImplicitCastExpr [[ADDR_80:0x[a-z0-9]*]] <line:16:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | |       `-DeclRefExpr [[ADDR_11]] <col:1> 'int ({{.*}})' Function [[ADDR_12]] 'also_after2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT:         | `-PseudoObjectExpr [[ADDR_81:0x[a-z0-9]*]] <line:43:42, col:54> 'int'
// C-NEXT:         |   |-CallExpr [[ADDR_82:0x[a-z0-9]*]] <col:42, col:54> 'int'
// C-NEXT:         |   | `-ImplicitCastExpr [[ADDR_83:0x[a-z0-9]*]] <col:42> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         |   |   `-DeclRefExpr [[ADDR_84:0x[a-z0-9]*]] <col:42> 'int ({{.*}})' Function [[ADDR_49]] 'also_after3' 'int ({{.*}})'
// C-NEXT:         |   `-CallExpr [[ADDR_85:0x[a-z0-9]*]] <line:19:1, line:43:54> 'int'
// C-NEXT:         |     `-ImplicitCastExpr [[ADDR_86:0x[a-z0-9]*]] <line:19:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         |       `-DeclRefExpr [[ADDR_19]] <col:1> 'int ({{.*}})' Function [[ADDR_20]] 'also_after3[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT:         `-PseudoObjectExpr [[ADDR_87:0x[a-z0-9]*]] <line:43:58, col:70> 'int'
// C-NEXT:           |-CallExpr [[ADDR_88:0x[a-z0-9]*]] <col:58, col:70> 'int'
// C-NEXT:           | `-ImplicitCastExpr [[ADDR_89:0x[a-z0-9]*]] <col:58> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:           |   `-DeclRefExpr [[ADDR_90:0x[a-z0-9]*]] <col:58> 'int ({{.*}})' Function [[ADDR_55]] 'also_after4' 'int ({{.*}})'
// C-NEXT:           `-CallExpr [[ADDR_91:0x[a-z0-9]*]] <line:22:1, line:43:70> 'int'
// C-NEXT:             `-ImplicitCastExpr [[ADDR_92:0x[a-z0-9]*]] <line:22:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:               `-DeclRefExpr [[ADDR_30]] <col:1> 'int ({{.*}})' Function [[ADDR_31]] 'also_after4[implementation={vendor(llvm)}]' 'int ({{.*}})'

// CXX:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:13:27> col:11 implicit used constexpr also_after1 'int ({{.*}})'
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_1:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_2:0x[a-z0-9]*]] <line:7:15> 'int ({{.*}})' Function [[ADDR_3:0x[a-z0-9]*]] 'also_after1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_3]] <col:15, line:15:1> line:7:15 constexpr also_after1[implementation={vendor(llvm)}] 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_4:0x[a-z0-9]*]] <line:13:29, line:15:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_5:0x[a-z0-9]*]] <line:14:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_6:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:16:1, col:28> col:12 implicit used also_after2 'int ({{.*}})' static
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_10:0x[a-z0-9]*]] 'also_after2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:18:1> line:16:1 also_after2[implementation={vendor(llvm)}] 'int ({{.*}})' static
// CXX-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:30, line:18:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:17:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_14:0x[a-z0-9]*]] <line:19:1, col:46> col:30 implicit used also_after3 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_15:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_16:0x[a-z0-9]*]] <col:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_17:0x[a-z0-9]*]] 'also_after3[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT: |-FunctionDecl [[ADDR_17]] <col:1, line:21:1> line:19:1 also_after3[implementation={vendor(llvm)}] 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT: | `-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:48, line:21:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_19:0x[a-z0-9]*]] <line:20:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_20:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: |-FunctionDecl [[ADDR_21:0x[a-z0-9]*]] <line:22:1, col:85> col:69 implicit used constexpr also_after4 'int ({{.*}}) __attribute__((nothrow))' static inline
// CXX-NEXT: | |-AlwaysInlineAttr [[ADDR_22:0x[a-z0-9]*]] <col:38> always_inline
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_23:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_24:0x[a-z0-9]*]] <col:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_25:0x[a-z0-9]*]] 'also_after4[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT: |-FunctionDecl [[ADDR_25]] <col:1, line:24:1> line:22:1 constexpr also_after4[implementation={vendor(llvm)}] 'int ({{.*}}) __attribute__((nothrow))' static inline
// CXX-NEXT: | |-CompoundStmt [[ADDR_26:0x[a-z0-9]*]] <col:87, line:24:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_27:0x[a-z0-9]*]] <line:23:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: | `-AlwaysInlineAttr [[ADDR_29:0x[a-z0-9]*]] <line:22:38> always_inline
// CXX-NEXT: |-FunctionDecl [[ADDR_30:0x[a-z0-9]*]] <line:27:1, line:29:1> line:27:5 invalid also_after1 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_31:0x[a-z0-9]*]] <col:23, line:29:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_32:0x[a-z0-9]*]] <line:28:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_33:0x[a-z0-9]*]] <col:10> 'int' 1
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_34:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_2]] <line:7:15> 'int ({{.*}})' Function [[ADDR_3]] 'also_after1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_35:0x[a-z0-9]*]] prev [[ADDR_7]] <line:30:1, line:32:1> line:30:5 used also_after2 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_36:0x[a-z0-9]*]] <col:23, line:32:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_37:0x[a-z0-9]*]] <line:31:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_38:0x[a-z0-9]*]] <col:10> 'int' 2
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_39:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:16:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_40:0x[a-z0-9]*]] prev [[ADDR_14]] <line:33:1, line:35:1> line:33:5 used also_after3 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_41:0x[a-z0-9]*]] <col:23, line:35:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_42:0x[a-z0-9]*]] <line:34:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_43:0x[a-z0-9]*]] <col:10> 'int' 3
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_44:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_16]] <line:19:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_17]] 'also_after3[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT: |-FunctionDecl [[ADDR_45:0x[a-z0-9]*]] <line:36:1, line:38:1> line:36:5 invalid also_after4 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_46:0x[a-z0-9]*]] <col:23, line:38:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_47:0x[a-z0-9]*]] <line:37:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_48:0x[a-z0-9]*]] <col:10> 'int' 4
// CXX-NEXT: | |-AlwaysInlineAttr [[ADDR_49:0x[a-z0-9]*]] <line:22:38> Inherited always_inline
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_50:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_24]] <col:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_25]] 'also_after4[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT: `-FunctionDecl [[ADDR_51:0x[a-z0-9]*]] <line:41:1, line:44:1> line:41:5 main 'int ({{.*}})'
// CXX-NEXT:   `-CompoundStmt [[ADDR_52:0x[a-z0-9]*]] <col:16, line:44:1>
// CXX-NEXT:     `-ReturnStmt [[ADDR_53:0x[a-z0-9]*]] <line:43:3, col:70>
// CXX-NEXT:       `-BinaryOperator [[ADDR_54:0x[a-z0-9]*]] <col:10, col:70> 'int' '+'
// CXX-NEXT:         |-BinaryOperator [[ADDR_55:0x[a-z0-9]*]] <col:10, col:54> 'int' '+'
// CXX-NEXT:         | |-BinaryOperator [[ADDR_56:0x[a-z0-9]*]] <col:10, col:38> 'int' '+'
// CXX-NEXT:         | | |-PseudoObjectExpr [[ADDR_57:0x[a-z0-9]*]] <col:10, col:22> 'int'
// CXX-NEXT:         | | | |-CallExpr [[ADDR_58:0x[a-z0-9]*]] <col:10, col:22> 'int'
// CXX-NEXT:         | | | | `-ImplicitCastExpr [[ADDR_59:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | | | |   `-DeclRefExpr [[ADDR_60:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_after1' 'int ({{.*}})'
// CXX-NEXT:         | | | `-CallExpr [[ADDR_61:0x[a-z0-9]*]] <line:7:15, line:43:22> 'int'
// CXX-NEXT:         | | |   `-ImplicitCastExpr [[ADDR_62:0x[a-z0-9]*]] <line:7:15> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | | |     `-DeclRefExpr [[ADDR_2]] <col:15> 'int ({{.*}})' Function [[ADDR_3]] 'also_after1[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT:         | | `-PseudoObjectExpr [[ADDR_63:0x[a-z0-9]*]] <line:43:26, col:38> 'int'
// CXX-NEXT:         | |   |-CallExpr [[ADDR_64:0x[a-z0-9]*]] <col:26, col:38> 'int'
// CXX-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_65:0x[a-z0-9]*]] <col:26> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | |   |   `-DeclRefExpr [[ADDR_66:0x[a-z0-9]*]] <col:26> 'int ({{.*}})' {{.*}}Function [[ADDR_35]] 'also_after2' 'int ({{.*}})'
// CXX-NEXT:         | |   `-CallExpr [[ADDR_67:0x[a-z0-9]*]] <line:16:1, line:43:38> 'int'
// CXX-NEXT:         | |     `-ImplicitCastExpr [[ADDR_68:0x[a-z0-9]*]] <line:16:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | |       `-DeclRefExpr [[ADDR_9]] <col:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after2[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT:         | `-PseudoObjectExpr [[ADDR_69:0x[a-z0-9]*]] <line:43:42, col:54> 'int'
// CXX-NEXT:         |   |-CallExpr [[ADDR_70:0x[a-z0-9]*]] <col:42, col:54> 'int'
// CXX-NEXT:         |   | `-ImplicitCastExpr [[ADDR_71:0x[a-z0-9]*]] <col:42> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         |   |   `-DeclRefExpr [[ADDR_72:0x[a-z0-9]*]] <col:42> 'int ({{.*}})' {{.*}}Function [[ADDR_40]] 'also_after3' 'int ({{.*}})'
// CXX-NEXT:         |   `-CallExpr [[ADDR_73:0x[a-z0-9]*]] <line:19:1, line:43:54> 'int'
// CXX-NEXT:         |     `-ImplicitCastExpr [[ADDR_74:0x[a-z0-9]*]] <line:19:1> 'int (*)({{.*}}) __attribute__((nothrow))' <FunctionToPointerDecay>
// CXX-NEXT:         |       `-DeclRefExpr [[ADDR_16]] <col:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_17]] 'also_after3[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT:         `-PseudoObjectExpr [[ADDR_75:0x[a-z0-9]*]] <line:43:58, col:70> 'int'
// CXX-NEXT:           |-CallExpr [[ADDR_76:0x[a-z0-9]*]] <col:58, col:70> 'int'
// CXX-NEXT:           | `-ImplicitCastExpr [[ADDR_77:0x[a-z0-9]*]] <col:58> 'int (*)({{.*}}) __attribute__((nothrow))' <FunctionToPointerDecay>
// CXX-NEXT:           |   `-DeclRefExpr [[ADDR_78:0x[a-z0-9]*]] <col:58> 'int ({{.*}}) __attribute__((nothrow))' {{.*}}Function [[ADDR_21]] 'also_after4' 'int ({{.*}}) __attribute__((nothrow))'
// CXX-NEXT:           `-CallExpr [[ADDR_79:0x[a-z0-9]*]] <line:22:1, line:43:70> 'int'
// CXX-NEXT:             `-ImplicitCastExpr [[ADDR_80:0x[a-z0-9]*]] <line:22:1> 'int (*)({{.*}}) __attribute__((nothrow))' <FunctionToPointerDecay>
// CXX-NEXT:               `-DeclRefExpr [[ADDR_24]] <col:1> 'int ({{.*}}) __attribute__((nothrow))' Function [[ADDR_25]] 'also_after4[implementation={vendor(llvm)}]' 'int ({{.*}}) __attribute__((nothrow))'
