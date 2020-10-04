// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s        -DUSE_FLOAT | FileCheck %s --check-prefix=C_FLOAT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++ -DUSE_FLOAT | FileCheck %s --check-prefix=CXX_FLOAT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s                    | FileCheck %s --check-prefix=C_INT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++             | FileCheck %s --check-prefix=CXX_INT
// expected-no-diagnostics

#ifdef __cplusplus
#define OVERLOADABLE
#else
#define OVERLOADABLE __attribute__((overloadable))
#endif

#ifdef USE_FLOAT
#define RETURN_TY float
#define BEFORE_BASE_RETURN_VALUE 0
#define BEFORE_VARIANT_RETURN_VALUE 1
#define AFTER__BASE_RETURN_VALUE 1
#define AFTER__VARIANT_RETURN_VALUE 0
#else
#define RETURN_TY int
#define BEFORE_BASE_RETURN_VALUE 1
#define BEFORE_VARIANT_RETURN_VALUE 0
#define AFTER__BASE_RETURN_VALUE 0
#define AFTER__VARIANT_RETURN_VALUE 1
#endif

OVERLOADABLE
RETURN_TY also_before(void) {
  return BEFORE_BASE_RETURN_VALUE;
}
OVERLOADABLE
RETURN_TY also_before(int i) {
  return BEFORE_BASE_RETURN_VALUE;
}

#pragma omp begin declare variant match(implementation = {extension(disable_implicit_base)})
OVERLOADABLE
int also_before(void) {
  return BEFORE_VARIANT_RETURN_VALUE;
}
OVERLOADABLE
int also_before(int i) {
  return BEFORE_VARIANT_RETURN_VALUE;
}

OVERLOADABLE
int also_after(double d) {
  return AFTER__VARIANT_RETURN_VALUE;
}
OVERLOADABLE
int also_after(long l) {
  return AFTER__VARIANT_RETURN_VALUE;
}
#pragma omp end declare variant

OVERLOADABLE
RETURN_TY also_after(double d) {
  return AFTER__BASE_RETURN_VALUE;
}
OVERLOADABLE
RETURN_TY also_after(long l) {
  return AFTER__BASE_RETURN_VALUE;
}

int main() {
  // Should return 0.
  return also_before() + also_before(1) + also_before(2.0f) + also_after(3.0) + also_after(4L);
}

// Make sure we see base calls in the FLOAT versions, that is no
// PseudoObjectExpr in those. In the INT versions we want PseudoObjectExpr (=
// variant calls) for the `*_before` functions but not the `*_after` ones
// (first 3 vs 2 last ones).

// C_FLOAT:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:30:1> line:28:11 used also_before 'float ({{.*}})'
// C_FLOAT-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:29, line:30:1>
// C_FLOAT-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:29:3, line:15:34>
// C_FLOAT-NEXT: | |   `-ImplicitCastExpr [[ADDR_3:0x[a-z0-9]*]] <col:34> 'float' <IntegralToFloating>
// C_FLOAT-NEXT: | |     `-IntegerLiteral [[ADDR_4:0x[a-z0-9]*]] <col:34> 'int' 0
// C_FLOAT-NEXT: | `-OverloadableAttr [[ADDR_5:0x[a-z0-9]*]] <line:10:37>
// C_FLOAT-NEXT: |-FunctionDecl [[ADDR_6:0x[a-z0-9]*]] <col:22, line:34:1> line:32:11 used also_before 'float (int)'
// C_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_7:0x[a-z0-9]*]] <col:23, col:27> col:27 i 'int'
// C_FLOAT-NEXT: | |-CompoundStmt [[ADDR_8:0x[a-z0-9]*]] <col:30, line:34:1>
// C_FLOAT-NEXT: | | `-ReturnStmt [[ADDR_9:0x[a-z0-9]*]] <line:33:3, line:15:34>
// C_FLOAT-NEXT: | |   `-ImplicitCastExpr [[ADDR_10:0x[a-z0-9]*]] <col:34> 'float' <IntegralToFloating>
// C_FLOAT-NEXT: | |     `-IntegerLiteral [[ADDR_11:0x[a-z0-9]*]] <col:34> 'int' 0
// C_FLOAT-NEXT: | `-OverloadableAttr [[ADDR_12:0x[a-z0-9]*]] <line:10:37>
// C_FLOAT-NEXT: |-FunctionDecl [[ADDR_13:0x[a-z0-9]*]] <col:22, line:40:1> line:10:22 also_before[implementation={extension(disable_implicit_base)}] 'int ({{.*}})'
// C_FLOAT-NEXT: | |-CompoundStmt [[ADDR_14:0x[a-z0-9]*]] <line:38:23, line:40:1>
// C_FLOAT-NEXT: | | `-ReturnStmt [[ADDR_15:0x[a-z0-9]*]] <line:39:3, line:16:37>
// C_FLOAT-NEXT: | |   `-IntegerLiteral [[ADDR_16:0x[a-z0-9]*]] <col:37> 'int' 1
// C_FLOAT-NEXT: | `-OverloadableAttr [[ADDR_17:0x[a-z0-9]*]] <line:10:37>
// C_FLOAT-NEXT: |-FunctionDecl [[ADDR_18:0x[a-z0-9]*]] <col:22, line:44:1> line:10:22 also_before[implementation={extension(disable_implicit_base)}] 'int (int)'
// C_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_19:0x[a-z0-9]*]] <line:42:17, col:21> col:21 i 'int'
// C_FLOAT-NEXT: | |-CompoundStmt [[ADDR_20:0x[a-z0-9]*]] <col:24, line:44:1>
// C_FLOAT-NEXT: | | `-ReturnStmt [[ADDR_21:0x[a-z0-9]*]] <line:43:3, line:16:37>
// C_FLOAT-NEXT: | |   `-IntegerLiteral [[ADDR_22:0x[a-z0-9]*]] <col:37> 'int' 1
// C_FLOAT-NEXT: | `-OverloadableAttr [[ADDR_23:0x[a-z0-9]*]] <line:10:37>
// C_FLOAT-NEXT: |-FunctionDecl [[ADDR_24:0x[a-z0-9]*]] <col:22, line:49:1> line:10:22 also_after[implementation={extension(disable_implicit_base)}] 'int (double)'
// C_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_25:0x[a-z0-9]*]] <line:47:16, col:23> col:23 d 'double'
// C_FLOAT-NEXT: | |-CompoundStmt [[ADDR_26:0x[a-z0-9]*]] <col:26, line:49:1>
// C_FLOAT-NEXT: | | `-ReturnStmt [[ADDR_27:0x[a-z0-9]*]] <line:48:3, line:18:37>
// C_FLOAT-NEXT: | |   `-IntegerLiteral [[ADDR_28:0x[a-z0-9]*]] <col:37> 'int' 0
// C_FLOAT-NEXT: | `-OverloadableAttr [[ADDR_29:0x[a-z0-9]*]] <line:10:37>
// C_FLOAT-NEXT: |-FunctionDecl [[ADDR_30:0x[a-z0-9]*]] <col:22, line:53:1> line:10:22 also_after[implementation={extension(disable_implicit_base)}] 'int (long)'
// C_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_31:0x[a-z0-9]*]] <line:51:16, col:21> col:21 l 'long'
// C_FLOAT-NEXT: | |-CompoundStmt [[ADDR_32:0x[a-z0-9]*]] <col:24, line:53:1>
// C_FLOAT-NEXT: | | `-ReturnStmt [[ADDR_33:0x[a-z0-9]*]] <line:52:3, line:18:37>
// C_FLOAT-NEXT: | |   `-IntegerLiteral [[ADDR_34:0x[a-z0-9]*]] <col:37> 'int' 0
// C_FLOAT-NEXT: | `-OverloadableAttr [[ADDR_35:0x[a-z0-9]*]] <line:10:37>
// C_FLOAT-NEXT: |-FunctionDecl [[ADDR_36:0x[a-z0-9]*]] <col:22, line:59:1> line:57:11 used also_after 'float (double)'
// C_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_37:0x[a-z0-9]*]] <col:22, col:29> col:29 d 'double'
// C_FLOAT-NEXT: | |-CompoundStmt [[ADDR_38:0x[a-z0-9]*]] <col:32, line:59:1>
// C_FLOAT-NEXT: | | `-ReturnStmt [[ADDR_39:0x[a-z0-9]*]] <line:58:3, line:17:34>
// C_FLOAT-NEXT: | |   `-ImplicitCastExpr [[ADDR_40:0x[a-z0-9]*]] <col:34> 'float' <IntegralToFloating>
// C_FLOAT-NEXT: | |     `-IntegerLiteral [[ADDR_41:0x[a-z0-9]*]] <col:34> 'int' 1
// C_FLOAT-NEXT: | `-OverloadableAttr [[ADDR_42:0x[a-z0-9]*]] <line:10:37>
// C_FLOAT-NEXT: |-FunctionDecl [[ADDR_43:0x[a-z0-9]*]] <col:22, line:63:1> line:61:11 used also_after 'float (long)'
// C_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_44:0x[a-z0-9]*]] <col:22, col:27> col:27 l 'long'
// C_FLOAT-NEXT: | |-CompoundStmt [[ADDR_45:0x[a-z0-9]*]] <col:30, line:63:1>
// C_FLOAT-NEXT: | | `-ReturnStmt [[ADDR_46:0x[a-z0-9]*]] <line:62:3, line:17:34>
// C_FLOAT-NEXT: | |   `-ImplicitCastExpr [[ADDR_47:0x[a-z0-9]*]] <col:34> 'float' <IntegralToFloating>
// C_FLOAT-NEXT: | |     `-IntegerLiteral [[ADDR_48:0x[a-z0-9]*]] <col:34> 'int' 1
// C_FLOAT-NEXT: | `-OverloadableAttr [[ADDR_49:0x[a-z0-9]*]] <line:10:37>
// C_FLOAT-NEXT: `-FunctionDecl [[ADDR_50:0x[a-z0-9]*]] <line:65:1, line:68:1> line:65:5 main 'int ({{.*}})'
// C_FLOAT-NEXT:   `-CompoundStmt [[ADDR_51:0x[a-z0-9]*]] <col:12, line:68:1>
// C_FLOAT-NEXT:     `-ReturnStmt [[ADDR_52:0x[a-z0-9]*]] <line:67:3, col:94>
// C_FLOAT-NEXT:       `-ImplicitCastExpr [[ADDR_53:0x[a-z0-9]*]] <col:10, col:94> 'int' <FloatingToIntegral>
// C_FLOAT-NEXT:         `-BinaryOperator [[ADDR_54:0x[a-z0-9]*]] <col:10, col:94> 'float' '+'
// C_FLOAT-NEXT:           |-BinaryOperator [[ADDR_55:0x[a-z0-9]*]] <col:10, col:77> 'float' '+'
// C_FLOAT-NEXT:           | |-BinaryOperator [[ADDR_56:0x[a-z0-9]*]] <col:10, col:59> 'float' '+'
// C_FLOAT-NEXT:           | | |-BinaryOperator [[ADDR_57:0x[a-z0-9]*]] <col:10, col:39> 'float' '+'
// C_FLOAT-NEXT:           | | | |-CallExpr [[ADDR_58:0x[a-z0-9]*]] <col:10, col:22> 'float'
// C_FLOAT-NEXT:           | | | | `-ImplicitCastExpr [[ADDR_59:0x[a-z0-9]*]] <col:10> 'float (*)({{.*}})' <FunctionToPointerDecay>
// C_FLOAT-NEXT:           | | | |   `-DeclRefExpr [[ADDR_60:0x[a-z0-9]*]] <col:10> 'float ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'float ({{.*}})'
// C_FLOAT-NEXT:           | | | `-CallExpr [[ADDR_61:0x[a-z0-9]*]] <col:26, col:39> 'float'
// C_FLOAT-NEXT:           | | |   |-ImplicitCastExpr [[ADDR_62:0x[a-z0-9]*]] <col:26> 'float (*)(int)' <FunctionToPointerDecay>
// C_FLOAT-NEXT:           | | |   | `-DeclRefExpr [[ADDR_63:0x[a-z0-9]*]] <col:26> 'float (int)' {{.*}}Function [[ADDR_6]] 'also_before' 'float (int)'
// C_FLOAT-NEXT:           | | |   `-IntegerLiteral [[ADDR_64:0x[a-z0-9]*]] <col:38> 'int' 1
// C_FLOAT-NEXT:           | | `-CallExpr [[ADDR_65:0x[a-z0-9]*]] <col:43, col:59> 'float'
// C_FLOAT-NEXT:           | |   |-ImplicitCastExpr [[ADDR_66:0x[a-z0-9]*]] <col:43> 'float (*)(int)' <FunctionToPointerDecay>
// C_FLOAT-NEXT:           | |   | `-DeclRefExpr [[ADDR_67:0x[a-z0-9]*]] <col:43> 'float (int)' {{.*}}Function [[ADDR_6]] 'also_before' 'float (int)'
// C_FLOAT-NEXT:           | |   `-ImplicitCastExpr [[ADDR_68:0x[a-z0-9]*]] <col:55> 'int' <FloatingToIntegral>
// C_FLOAT-NEXT:           | |     `-FloatingLiteral [[ADDR_69:0x[a-z0-9]*]] <col:55> 'float' 2.000000e+00
// C_FLOAT-NEXT:           | `-CallExpr [[ADDR_70:0x[a-z0-9]*]] <col:63, col:77> 'float'
// C_FLOAT-NEXT:           |   |-ImplicitCastExpr [[ADDR_71:0x[a-z0-9]*]] <col:63> 'float (*)(double)' <FunctionToPointerDecay>
// C_FLOAT-NEXT:           |   | `-DeclRefExpr [[ADDR_72:0x[a-z0-9]*]] <col:63> 'float (double)' {{.*}}Function [[ADDR_36]] 'also_after' 'float (double)'
// C_FLOAT-NEXT:           |   `-FloatingLiteral [[ADDR_73:0x[a-z0-9]*]] <col:74> 'double' 3.000000e+00
// C_FLOAT-NEXT:           `-CallExpr [[ADDR_74:0x[a-z0-9]*]] <col:81, col:94> 'float'
// C_FLOAT-NEXT:             |-ImplicitCastExpr [[ADDR_75:0x[a-z0-9]*]] <col:81> 'float (*)(long)' <FunctionToPointerDecay>
// C_FLOAT-NEXT:             | `-DeclRefExpr [[ADDR_76:0x[a-z0-9]*]] <col:81> 'float (long)' {{.*}}Function [[ADDR_43]] 'also_after' 'float (long)'
// C_FLOAT-NEXT:             `-IntegerLiteral [[ADDR_77:0x[a-z0-9]*]] <col:92> 'long' 4

// CXX_FLOAT:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:30:1> line:28:11 used also_before 'float ({{.*}})'
// CXX_FLOAT-NEXT: | `-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:29, line:30:1>
// CXX_FLOAT-NEXT: |   `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:29:3, line:15:34>
// CXX_FLOAT-NEXT: |     `-ImplicitCastExpr [[ADDR_3:0x[a-z0-9]*]] <col:34> 'float' <IntegralToFloating>
// CXX_FLOAT-NEXT: |       `-IntegerLiteral [[ADDR_4:0x[a-z0-9]*]] <col:34> 'int' 0
// CXX_FLOAT-NEXT: |-FunctionDecl [[ADDR_5:0x[a-z0-9]*]] <line:14:19, line:34:1> line:32:11 used also_before 'float (int)'
// CXX_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_6:0x[a-z0-9]*]] <col:23, col:27> col:27 i 'int'
// CXX_FLOAT-NEXT: | `-CompoundStmt [[ADDR_7:0x[a-z0-9]*]] <col:30, line:34:1>
// CXX_FLOAT-NEXT: |   `-ReturnStmt [[ADDR_8:0x[a-z0-9]*]] <line:33:3, line:15:34>
// CXX_FLOAT-NEXT: |     `-ImplicitCastExpr [[ADDR_9:0x[a-z0-9]*]] <col:34> 'float' <IntegralToFloating>
// CXX_FLOAT-NEXT: |       `-IntegerLiteral [[ADDR_10:0x[a-z0-9]*]] <col:34> 'int' 0
// CXX_FLOAT-NEXT: |-FunctionDecl [[ADDR_11:0x[a-z0-9]*]] <line:38:1, line:40:1> line:38:1 also_before[implementation={extension(disable_implicit_base)}] 'int ({{.*}})'
// CXX_FLOAT-NEXT: | `-CompoundStmt [[ADDR_12:0x[a-z0-9]*]] <col:23, line:40:1>
// CXX_FLOAT-NEXT: |   `-ReturnStmt [[ADDR_13:0x[a-z0-9]*]] <line:39:3, line:16:37>
// CXX_FLOAT-NEXT: |     `-IntegerLiteral [[ADDR_14:0x[a-z0-9]*]] <col:37> 'int' 1
// CXX_FLOAT-NEXT: |-FunctionDecl [[ADDR_15:0x[a-z0-9]*]] <line:42:1, line:44:1> line:42:1 also_before[implementation={extension(disable_implicit_base)}] 'int (int)'
// CXX_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_16:0x[a-z0-9]*]] <col:17, col:21> col:21 i 'int'
// CXX_FLOAT-NEXT: | `-CompoundStmt [[ADDR_17:0x[a-z0-9]*]] <col:24, line:44:1>
// CXX_FLOAT-NEXT: |   `-ReturnStmt [[ADDR_18:0x[a-z0-9]*]] <line:43:3, line:16:37>
// CXX_FLOAT-NEXT: |     `-IntegerLiteral [[ADDR_19:0x[a-z0-9]*]] <col:37> 'int' 1
// CXX_FLOAT-NEXT: |-FunctionDecl [[ADDR_20:0x[a-z0-9]*]] <line:47:1, line:49:1> line:47:1 also_after[implementation={extension(disable_implicit_base)}] 'int (double)'
// CXX_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_21:0x[a-z0-9]*]] <col:16, col:23> col:23 d 'double'
// CXX_FLOAT-NEXT: | `-CompoundStmt [[ADDR_22:0x[a-z0-9]*]] <col:26, line:49:1>
// CXX_FLOAT-NEXT: |   `-ReturnStmt [[ADDR_23:0x[a-z0-9]*]] <line:48:3, line:18:37>
// CXX_FLOAT-NEXT: |     `-IntegerLiteral [[ADDR_24:0x[a-z0-9]*]] <col:37> 'int' 0
// CXX_FLOAT-NEXT: |-FunctionDecl [[ADDR_25:0x[a-z0-9]*]] <line:51:1, line:53:1> line:51:1 also_after[implementation={extension(disable_implicit_base)}] 'int (long)'
// CXX_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_26:0x[a-z0-9]*]] <col:16, col:21> col:21 l 'long'
// CXX_FLOAT-NEXT: | `-CompoundStmt [[ADDR_27:0x[a-z0-9]*]] <col:24, line:53:1>
// CXX_FLOAT-NEXT: |   `-ReturnStmt [[ADDR_28:0x[a-z0-9]*]] <line:52:3, line:18:37>
// CXX_FLOAT-NEXT: |     `-IntegerLiteral [[ADDR_29:0x[a-z0-9]*]] <col:37> 'int' 0
// CXX_FLOAT-NEXT: |-FunctionDecl [[ADDR_30:0x[a-z0-9]*]] <line:14:19, line:59:1> line:57:11 used also_after 'float (double)'
// CXX_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_31:0x[a-z0-9]*]] <col:22, col:29> col:29 d 'double'
// CXX_FLOAT-NEXT: | `-CompoundStmt [[ADDR_32:0x[a-z0-9]*]] <col:32, line:59:1>
// CXX_FLOAT-NEXT: |   `-ReturnStmt [[ADDR_33:0x[a-z0-9]*]] <line:58:3, line:17:34>
// CXX_FLOAT-NEXT: |     `-ImplicitCastExpr [[ADDR_34:0x[a-z0-9]*]] <col:34> 'float' <IntegralToFloating>
// CXX_FLOAT-NEXT: |       `-IntegerLiteral [[ADDR_35:0x[a-z0-9]*]] <col:34> 'int' 1
// CXX_FLOAT-NEXT: |-FunctionDecl [[ADDR_36:0x[a-z0-9]*]] <line:14:19, line:63:1> line:61:11 used also_after 'float (long)'
// CXX_FLOAT-NEXT: | |-ParmVarDecl [[ADDR_37:0x[a-z0-9]*]] <col:22, col:27> col:27 l 'long'
// CXX_FLOAT-NEXT: | `-CompoundStmt [[ADDR_38:0x[a-z0-9]*]] <col:30, line:63:1>
// CXX_FLOAT-NEXT: |   `-ReturnStmt [[ADDR_39:0x[a-z0-9]*]] <line:62:3, line:17:34>
// CXX_FLOAT-NEXT: |     `-ImplicitCastExpr [[ADDR_40:0x[a-z0-9]*]] <col:34> 'float' <IntegralToFloating>
// CXX_FLOAT-NEXT: |       `-IntegerLiteral [[ADDR_41:0x[a-z0-9]*]] <col:34> 'int' 1
// CXX_FLOAT-NEXT: `-FunctionDecl [[ADDR_42:0x[a-z0-9]*]] <line:65:1, line:68:1> line:65:5 main 'int ({{.*}})'
// CXX_FLOAT-NEXT:   `-CompoundStmt [[ADDR_43:0x[a-z0-9]*]] <col:12, line:68:1>
// CXX_FLOAT-NEXT:     `-ReturnStmt [[ADDR_44:0x[a-z0-9]*]] <line:67:3, col:94>
// CXX_FLOAT-NEXT:       `-ImplicitCastExpr [[ADDR_45:0x[a-z0-9]*]] <col:10, col:94> 'int' <FloatingToIntegral>
// CXX_FLOAT-NEXT:         `-BinaryOperator [[ADDR_46:0x[a-z0-9]*]] <col:10, col:94> 'float' '+'
// CXX_FLOAT-NEXT:           |-BinaryOperator [[ADDR_47:0x[a-z0-9]*]] <col:10, col:77> 'float' '+'
// CXX_FLOAT-NEXT:           | |-BinaryOperator [[ADDR_48:0x[a-z0-9]*]] <col:10, col:59> 'float' '+'
// CXX_FLOAT-NEXT:           | | |-BinaryOperator [[ADDR_49:0x[a-z0-9]*]] <col:10, col:39> 'float' '+'
// CXX_FLOAT-NEXT:           | | | |-CallExpr [[ADDR_50:0x[a-z0-9]*]] <col:10, col:22> 'float'
// CXX_FLOAT-NEXT:           | | | | `-ImplicitCastExpr [[ADDR_51:0x[a-z0-9]*]] <col:10> 'float (*)({{.*}})' <FunctionToPointerDecay>
// CXX_FLOAT-NEXT:           | | | |   `-DeclRefExpr [[ADDR_52:0x[a-z0-9]*]] <col:10> 'float ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'float ({{.*}})'
// CXX_FLOAT-NEXT:           | | | `-CallExpr [[ADDR_53:0x[a-z0-9]*]] <col:26, col:39> 'float'
// CXX_FLOAT-NEXT:           | | |   |-ImplicitCastExpr [[ADDR_54:0x[a-z0-9]*]] <col:26> 'float (*)(int)' <FunctionToPointerDecay>
// CXX_FLOAT-NEXT:           | | |   | `-DeclRefExpr [[ADDR_55:0x[a-z0-9]*]] <col:26> 'float (int)' {{.*}}Function [[ADDR_5]] 'also_before' 'float (int)'
// CXX_FLOAT-NEXT:           | | |   `-IntegerLiteral [[ADDR_56:0x[a-z0-9]*]] <col:38> 'int' 1
// CXX_FLOAT-NEXT:           | | `-CallExpr [[ADDR_57:0x[a-z0-9]*]] <col:43, col:59> 'float'
// CXX_FLOAT-NEXT:           | |   |-ImplicitCastExpr [[ADDR_58:0x[a-z0-9]*]] <col:43> 'float (*)(int)' <FunctionToPointerDecay>
// CXX_FLOAT-NEXT:           | |   | `-DeclRefExpr [[ADDR_59:0x[a-z0-9]*]] <col:43> 'float (int)' {{.*}}Function [[ADDR_5]] 'also_before' 'float (int)'
// CXX_FLOAT-NEXT:           | |   `-ImplicitCastExpr [[ADDR_60:0x[a-z0-9]*]] <col:55> 'int' <FloatingToIntegral>
// CXX_FLOAT-NEXT:           | |     `-FloatingLiteral [[ADDR_61:0x[a-z0-9]*]] <col:55> 'float' 2.000000e+00
// CXX_FLOAT-NEXT:           | `-CallExpr [[ADDR_62:0x[a-z0-9]*]] <col:63, col:77> 'float'
// CXX_FLOAT-NEXT:           |   |-ImplicitCastExpr [[ADDR_63:0x[a-z0-9]*]] <col:63> 'float (*)(double)' <FunctionToPointerDecay>
// CXX_FLOAT-NEXT:           |   | `-DeclRefExpr [[ADDR_64:0x[a-z0-9]*]] <col:63> 'float (double)' {{.*}}Function [[ADDR_30]] 'also_after' 'float (double)'
// CXX_FLOAT-NEXT:           |   `-FloatingLiteral [[ADDR_65:0x[a-z0-9]*]] <col:74> 'double' 3.000000e+00
// CXX_FLOAT-NEXT:           `-CallExpr [[ADDR_66:0x[a-z0-9]*]] <col:81, col:94> 'float'
// CXX_FLOAT-NEXT:             |-ImplicitCastExpr [[ADDR_67:0x[a-z0-9]*]] <col:81> 'float (*)(long)' <FunctionToPointerDecay>
// CXX_FLOAT-NEXT:             | `-DeclRefExpr [[ADDR_68:0x[a-z0-9]*]] <col:81> 'float (long)' {{.*}}Function [[ADDR_36]] 'also_after' 'float (long)'
// CXX_FLOAT-NEXT:             `-IntegerLiteral [[ADDR_69:0x[a-z0-9]*]] <col:92> 'long' 4

// C_INT:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:30:1> line:28:11 used also_before 'int ({{.*}})'
// C_INT-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:29, line:30:1>
// C_INT-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:29:3, line:21:34>
// C_INT-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:34> 'int' 1
// C_INT-NEXT: | |-OverloadableAttr [[ADDR_4:0x[a-z0-9]*]] <line:10:37>
// C_INT-NEXT: | `-OMPDeclareVariantAttr [[ADDR_5:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(disable_implicit_base)}
// C_INT-NEXT: |   `-DeclRefExpr [[ADDR_6:0x[a-z0-9]*]] <col:22> 'int ({{.*}})' Function [[ADDR_7:0x[a-z0-9]*]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int ({{.*}})'
// C_INT-NEXT: |-FunctionDecl [[ADDR_8:0x[a-z0-9]*]] <col:22, line:34:1> line:32:11 used also_before 'int (int)'
// C_INT-NEXT: | |-ParmVarDecl [[ADDR_9:0x[a-z0-9]*]] <col:23, col:27> col:27 i 'int'
// C_INT-NEXT: | |-CompoundStmt [[ADDR_10:0x[a-z0-9]*]] <col:30, line:34:1>
// C_INT-NEXT: | | `-ReturnStmt [[ADDR_11:0x[a-z0-9]*]] <line:33:3, line:21:34>
// C_INT-NEXT: | |   `-IntegerLiteral [[ADDR_12:0x[a-z0-9]*]] <col:34> 'int' 1
// C_INT-NEXT: | |-OverloadableAttr [[ADDR_13:0x[a-z0-9]*]] <line:10:37>
// C_INT-NEXT: | `-OMPDeclareVariantAttr [[ADDR_14:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(disable_implicit_base)}
// C_INT-NEXT: |   `-DeclRefExpr [[ADDR_15:0x[a-z0-9]*]] <col:22> 'int (int)' Function [[ADDR_16:0x[a-z0-9]*]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int (int)'
// C_INT-NEXT: |-FunctionDecl [[ADDR_7]] <col:22, line:40:1> line:10:22 also_before[implementation={extension(disable_implicit_base)}] 'int ({{.*}})'
// C_INT-NEXT: | |-CompoundStmt [[ADDR_17:0x[a-z0-9]*]] <line:38:23, line:40:1>
// C_INT-NEXT: | | `-ReturnStmt [[ADDR_18:0x[a-z0-9]*]] <line:39:3, line:22:37>
// C_INT-NEXT: | |   `-IntegerLiteral [[ADDR_19:0x[a-z0-9]*]] <col:37> 'int' 0
// C_INT-NEXT: | `-OverloadableAttr [[ADDR_20:0x[a-z0-9]*]] <line:10:37>
// C_INT-NEXT: |-FunctionDecl [[ADDR_16]] <col:22, line:44:1> line:10:22 also_before[implementation={extension(disable_implicit_base)}] 'int (int)'
// C_INT-NEXT: | |-ParmVarDecl [[ADDR_21:0x[a-z0-9]*]] <line:42:17, col:21> col:21 i 'int'
// C_INT-NEXT: | |-CompoundStmt [[ADDR_22:0x[a-z0-9]*]] <col:24, line:44:1>
// C_INT-NEXT: | | `-ReturnStmt [[ADDR_23:0x[a-z0-9]*]] <line:43:3, line:22:37>
// C_INT-NEXT: | |   `-IntegerLiteral [[ADDR_24:0x[a-z0-9]*]] <col:37> 'int' 0
// C_INT-NEXT: | `-OverloadableAttr [[ADDR_25:0x[a-z0-9]*]] <line:10:37>
// C_INT-NEXT: |-FunctionDecl [[ADDR_26:0x[a-z0-9]*]] <col:22, line:49:1> line:10:22 also_after[implementation={extension(disable_implicit_base)}] 'int (double)'
// C_INT-NEXT: | |-ParmVarDecl [[ADDR_27:0x[a-z0-9]*]] <line:47:16, col:23> col:23 d 'double'
// C_INT-NEXT: | |-CompoundStmt [[ADDR_28:0x[a-z0-9]*]] <col:26, line:49:1>
// C_INT-NEXT: | | `-ReturnStmt [[ADDR_29:0x[a-z0-9]*]] <line:48:3, line:24:37>
// C_INT-NEXT: | |   `-IntegerLiteral [[ADDR_30:0x[a-z0-9]*]] <col:37> 'int' 1
// C_INT-NEXT: | `-OverloadableAttr [[ADDR_31:0x[a-z0-9]*]] <line:10:37>
// C_INT-NEXT: |-FunctionDecl [[ADDR_32:0x[a-z0-9]*]] <col:22, line:53:1> line:10:22 also_after[implementation={extension(disable_implicit_base)}] 'int (long)'
// C_INT-NEXT: | |-ParmVarDecl [[ADDR_33:0x[a-z0-9]*]] <line:51:16, col:21> col:21 l 'long'
// C_INT-NEXT: | |-CompoundStmt [[ADDR_34:0x[a-z0-9]*]] <col:24, line:53:1>
// C_INT-NEXT: | | `-ReturnStmt [[ADDR_35:0x[a-z0-9]*]] <line:52:3, line:24:37>
// C_INT-NEXT: | |   `-IntegerLiteral [[ADDR_36:0x[a-z0-9]*]] <col:37> 'int' 1
// C_INT-NEXT: | `-OverloadableAttr [[ADDR_37:0x[a-z0-9]*]] <line:10:37>
// C_INT-NEXT: |-FunctionDecl [[ADDR_38:0x[a-z0-9]*]] <col:22, line:59:1> line:57:11 used also_after 'int (double)'
// C_INT-NEXT: | |-ParmVarDecl [[ADDR_39:0x[a-z0-9]*]] <col:22, col:29> col:29 d 'double'
// C_INT-NEXT: | |-CompoundStmt [[ADDR_40:0x[a-z0-9]*]] <col:32, line:59:1>
// C_INT-NEXT: | | `-ReturnStmt [[ADDR_41:0x[a-z0-9]*]] <line:58:3, line:23:34>
// C_INT-NEXT: | |   `-IntegerLiteral [[ADDR_42:0x[a-z0-9]*]] <col:34> 'int' 0
// C_INT-NEXT: | `-OverloadableAttr [[ADDR_43:0x[a-z0-9]*]] <line:10:37>
// C_INT-NEXT: |-FunctionDecl [[ADDR_44:0x[a-z0-9]*]] <col:22, line:63:1> line:61:11 used also_after 'int (long)'
// C_INT-NEXT: | |-ParmVarDecl [[ADDR_45:0x[a-z0-9]*]] <col:22, col:27> col:27 l 'long'
// C_INT-NEXT: | |-CompoundStmt [[ADDR_46:0x[a-z0-9]*]] <col:30, line:63:1>
// C_INT-NEXT: | | `-ReturnStmt [[ADDR_47:0x[a-z0-9]*]] <line:62:3, line:23:34>
// C_INT-NEXT: | |   `-IntegerLiteral [[ADDR_48:0x[a-z0-9]*]] <col:34> 'int' 0
// C_INT-NEXT: | `-OverloadableAttr [[ADDR_49:0x[a-z0-9]*]] <line:10:37>
// C_INT-NEXT: `-FunctionDecl [[ADDR_50:0x[a-z0-9]*]] <line:65:1, line:68:1> line:65:5 main 'int ({{.*}})'
// C_INT-NEXT:   `-CompoundStmt [[ADDR_51:0x[a-z0-9]*]] <col:12, line:68:1>
// C_INT-NEXT:     `-ReturnStmt [[ADDR_52:0x[a-z0-9]*]] <line:67:3, col:94>
// C_INT-NEXT:       `-BinaryOperator [[ADDR_53:0x[a-z0-9]*]] <col:10, col:94> 'int' '+'
// C_INT-NEXT:         |-BinaryOperator [[ADDR_54:0x[a-z0-9]*]] <col:10, col:77> 'int' '+'
// C_INT-NEXT:         | |-BinaryOperator [[ADDR_55:0x[a-z0-9]*]] <col:10, col:59> 'int' '+'
// C_INT-NEXT:         | | |-BinaryOperator [[ADDR_56:0x[a-z0-9]*]] <col:10, col:39> 'int' '+'
// C_INT-NEXT:         | | | |-PseudoObjectExpr [[ADDR_57:0x[a-z0-9]*]] <col:10, col:22> 'int'
// C_INT-NEXT:         | | | | |-CallExpr [[ADDR_58:0x[a-z0-9]*]] <col:10, col:22> 'int'
// C_INT-NEXT:         | | | | | `-ImplicitCastExpr [[ADDR_59:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C_INT-NEXT:         | | | | |   `-DeclRefExpr [[ADDR_60:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// C_INT-NEXT:         | | | | `-CallExpr [[ADDR_61:0x[a-z0-9]*]] <line:10:22, line:67:22> 'int'
// C_INT-NEXT:         | | | |   `-ImplicitCastExpr [[ADDR_62:0x[a-z0-9]*]] <line:10:22> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C_INT-NEXT:         | | | |     `-DeclRefExpr [[ADDR_6]] <col:22> 'int ({{.*}})' Function [[ADDR_7]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int ({{.*}})'
// C_INT-NEXT:         | | | `-PseudoObjectExpr [[ADDR_63:0x[a-z0-9]*]] <line:67:26, col:39> 'int'
// C_INT-NEXT:         | | |   |-CallExpr [[ADDR_64:0x[a-z0-9]*]] <col:26, col:39> 'int'
// C_INT-NEXT:         | | |   | |-ImplicitCastExpr [[ADDR_65:0x[a-z0-9]*]] <col:26> 'int (*)(int)' <FunctionToPointerDecay>
// C_INT-NEXT:         | | |   | | `-DeclRefExpr [[ADDR_66:0x[a-z0-9]*]] <col:26> 'int (int)' {{.*}}Function [[ADDR_8]] 'also_before' 'int (int)'
// C_INT-NEXT:         | | |   | `-IntegerLiteral [[ADDR_67:0x[a-z0-9]*]] <col:38> 'int' 1
// C_INT-NEXT:         | | |   `-CallExpr [[ADDR_68:0x[a-z0-9]*]] <line:10:22, line:67:39> 'int'
// C_INT-NEXT:         | | |     |-ImplicitCastExpr [[ADDR_69:0x[a-z0-9]*]] <line:10:22> 'int (*)(int)' <FunctionToPointerDecay>
// C_INT-NEXT:         | | |     | `-DeclRefExpr [[ADDR_15]] <col:22> 'int (int)' Function [[ADDR_16]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int (int)'
// C_INT-NEXT:         | | |     `-IntegerLiteral [[ADDR_67]] <line:67:38> 'int' 1
// C_INT-NEXT:         | | `-PseudoObjectExpr [[ADDR_70:0x[a-z0-9]*]] <col:43, col:59> 'int'
// C_INT-NEXT:         | |   |-CallExpr [[ADDR_71:0x[a-z0-9]*]] <col:43, col:59> 'int'
// C_INT-NEXT:         | |   | |-ImplicitCastExpr [[ADDR_72:0x[a-z0-9]*]] <col:43> 'int (*)(int)' <FunctionToPointerDecay>
// C_INT-NEXT:         | |   | | `-DeclRefExpr [[ADDR_73:0x[a-z0-9]*]] <col:43> 'int (int)' {{.*}}Function [[ADDR_8]] 'also_before' 'int (int)'
// C_INT-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_74:0x[a-z0-9]*]] <col:55> 'int' <FloatingToIntegral>
// C_INT-NEXT:         | |   |   `-FloatingLiteral [[ADDR_75:0x[a-z0-9]*]] <col:55> 'float' 2.000000e+00
// C_INT-NEXT:         | |   `-CallExpr [[ADDR_76:0x[a-z0-9]*]] <line:10:22, line:67:59> 'int'
// C_INT-NEXT:         | |     |-ImplicitCastExpr [[ADDR_77:0x[a-z0-9]*]] <line:10:22> 'int (*)(int)' <FunctionToPointerDecay>
// C_INT-NEXT:         | |     | `-DeclRefExpr [[ADDR_15]] <col:22> 'int (int)' Function [[ADDR_16]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int (int)'
// C_INT-NEXT:         | |     `-ImplicitCastExpr [[ADDR_78:0x[a-z0-9]*]] <line:67:55> 'int' <FloatingToIntegral>
// C_INT-NEXT:         | |       `-FloatingLiteral [[ADDR_75]] <col:55> 'float' 2.000000e+00
// C_INT-NEXT:         | `-CallExpr [[ADDR_79:0x[a-z0-9]*]] <col:63, col:77> 'int'
// C_INT-NEXT:         |   |-ImplicitCastExpr [[ADDR_80:0x[a-z0-9]*]] <col:63> 'int (*)(double)' <FunctionToPointerDecay>
// C_INT-NEXT:         |   | `-DeclRefExpr [[ADDR_81:0x[a-z0-9]*]] <col:63> 'int (double)' {{.*}}Function [[ADDR_38]] 'also_after' 'int (double)'
// C_INT-NEXT:         |   `-FloatingLiteral [[ADDR_82:0x[a-z0-9]*]] <col:74> 'double' 3.000000e+00
// C_INT-NEXT:         `-CallExpr [[ADDR_83:0x[a-z0-9]*]] <col:81, col:94> 'int'
// C_INT-NEXT:           |-ImplicitCastExpr [[ADDR_84:0x[a-z0-9]*]] <col:81> 'int (*)(long)' <FunctionToPointerDecay>
// C_INT-NEXT:           | `-DeclRefExpr [[ADDR_85:0x[a-z0-9]*]] <col:81> 'int (long)' {{.*}}Function [[ADDR_44]] 'also_after' 'int (long)'
// C_INT-NEXT:           `-IntegerLiteral [[ADDR_86:0x[a-z0-9]*]] <col:92> 'long' 4

// CXX_INT:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:30:1> line:28:11 used also_before 'int ({{.*}})'
// CXX_INT-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:29, line:30:1>
// CXX_INT-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:29:3, line:21:34>
// CXX_INT-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:34> 'int' 1
// CXX_INT-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(disable_implicit_base)}
// CXX_INT-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:38:1> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int ({{.*}})'
// CXX_INT-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:20:19, line:34:1> line:32:11 used also_before 'int (int)'
// CXX_INT-NEXT: | |-ParmVarDecl [[ADDR_8:0x[a-z0-9]*]] <col:23, col:27> col:27 i 'int'
// CXX_INT-NEXT: | |-CompoundStmt [[ADDR_9:0x[a-z0-9]*]] <col:30, line:34:1>
// CXX_INT-NEXT: | | `-ReturnStmt [[ADDR_10:0x[a-z0-9]*]] <line:33:3, line:21:34>
// CXX_INT-NEXT: | |   `-IntegerLiteral [[ADDR_11:0x[a-z0-9]*]] <col:34> 'int' 1
// CXX_INT-NEXT: | `-OMPDeclareVariantAttr [[ADDR_12:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(disable_implicit_base)}
// CXX_INT-NEXT: |   `-DeclRefExpr [[ADDR_13:0x[a-z0-9]*]] <line:42:1> 'int (int)' Function [[ADDR_14:0x[a-z0-9]*]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int (int)'
// CXX_INT-NEXT: |-FunctionDecl [[ADDR_6]] <line:38:1, line:40:1> line:38:1 also_before[implementation={extension(disable_implicit_base)}] 'int ({{.*}})'
// CXX_INT-NEXT: | `-CompoundStmt [[ADDR_15:0x[a-z0-9]*]] <col:23, line:40:1>
// CXX_INT-NEXT: |   `-ReturnStmt [[ADDR_16:0x[a-z0-9]*]] <line:39:3, line:22:37>
// CXX_INT-NEXT: |     `-IntegerLiteral [[ADDR_17:0x[a-z0-9]*]] <col:37> 'int' 0
// CXX_INT-NEXT: |-FunctionDecl [[ADDR_14]] <line:42:1, line:44:1> line:42:1 also_before[implementation={extension(disable_implicit_base)}] 'int (int)'
// CXX_INT-NEXT: | |-ParmVarDecl [[ADDR_18:0x[a-z0-9]*]] <col:17, col:21> col:21 i 'int'
// CXX_INT-NEXT: | `-CompoundStmt [[ADDR_19:0x[a-z0-9]*]] <col:24, line:44:1>
// CXX_INT-NEXT: |   `-ReturnStmt [[ADDR_20:0x[a-z0-9]*]] <line:43:3, line:22:37>
// CXX_INT-NEXT: |     `-IntegerLiteral [[ADDR_21:0x[a-z0-9]*]] <col:37> 'int' 0
// CXX_INT-NEXT: |-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:47:1, line:49:1> line:47:1 also_after[implementation={extension(disable_implicit_base)}] 'int (double)'
// CXX_INT-NEXT: | |-ParmVarDecl [[ADDR_23:0x[a-z0-9]*]] <col:16, col:23> col:23 d 'double'
// CXX_INT-NEXT: | `-CompoundStmt [[ADDR_24:0x[a-z0-9]*]] <col:26, line:49:1>
// CXX_INT-NEXT: |   `-ReturnStmt [[ADDR_25:0x[a-z0-9]*]] <line:48:3, line:24:37>
// CXX_INT-NEXT: |     `-IntegerLiteral [[ADDR_26:0x[a-z0-9]*]] <col:37> 'int' 1
// CXX_INT-NEXT: |-FunctionDecl [[ADDR_27:0x[a-z0-9]*]] <line:51:1, line:53:1> line:51:1 also_after[implementation={extension(disable_implicit_base)}] 'int (long)'
// CXX_INT-NEXT: | |-ParmVarDecl [[ADDR_28:0x[a-z0-9]*]] <col:16, col:21> col:21 l 'long'
// CXX_INT-NEXT: | `-CompoundStmt [[ADDR_29:0x[a-z0-9]*]] <col:24, line:53:1>
// CXX_INT-NEXT: |   `-ReturnStmt [[ADDR_30:0x[a-z0-9]*]] <line:52:3, line:24:37>
// CXX_INT-NEXT: |     `-IntegerLiteral [[ADDR_31:0x[a-z0-9]*]] <col:37> 'int' 1
// CXX_INT-NEXT: |-FunctionDecl [[ADDR_32:0x[a-z0-9]*]] <line:20:19, line:59:1> line:57:11 used also_after 'int (double)'
// CXX_INT-NEXT: | |-ParmVarDecl [[ADDR_33:0x[a-z0-9]*]] <col:22, col:29> col:29 d 'double'
// CXX_INT-NEXT: | `-CompoundStmt [[ADDR_34:0x[a-z0-9]*]] <col:32, line:59:1>
// CXX_INT-NEXT: |   `-ReturnStmt [[ADDR_35:0x[a-z0-9]*]] <line:58:3, line:23:34>
// CXX_INT-NEXT: |     `-IntegerLiteral [[ADDR_36:0x[a-z0-9]*]] <col:34> 'int' 0
// CXX_INT-NEXT: |-FunctionDecl [[ADDR_37:0x[a-z0-9]*]] <line:20:19, line:63:1> line:61:11 used also_after 'int (long)'
// CXX_INT-NEXT: | |-ParmVarDecl [[ADDR_38:0x[a-z0-9]*]] <col:22, col:27> col:27 l 'long'
// CXX_INT-NEXT: | `-CompoundStmt [[ADDR_39:0x[a-z0-9]*]] <col:30, line:63:1>
// CXX_INT-NEXT: |   `-ReturnStmt [[ADDR_40:0x[a-z0-9]*]] <line:62:3, line:23:34>
// CXX_INT-NEXT: |     `-IntegerLiteral [[ADDR_41:0x[a-z0-9]*]] <col:34> 'int' 0
// CXX_INT-NEXT: `-FunctionDecl [[ADDR_42:0x[a-z0-9]*]] <line:65:1, line:68:1> line:65:5 main 'int ({{.*}})'
// CXX_INT-NEXT:   `-CompoundStmt [[ADDR_43:0x[a-z0-9]*]] <col:12, line:68:1>
// CXX_INT-NEXT:     `-ReturnStmt [[ADDR_44:0x[a-z0-9]*]] <line:67:3, col:94>
// CXX_INT-NEXT:       `-BinaryOperator [[ADDR_45:0x[a-z0-9]*]] <col:10, col:94> 'int' '+'
// CXX_INT-NEXT:         |-BinaryOperator [[ADDR_46:0x[a-z0-9]*]] <col:10, col:77> 'int' '+'
// CXX_INT-NEXT:         | |-BinaryOperator [[ADDR_47:0x[a-z0-9]*]] <col:10, col:59> 'int' '+'
// CXX_INT-NEXT:         | | |-BinaryOperator [[ADDR_48:0x[a-z0-9]*]] <col:10, col:39> 'int' '+'
// CXX_INT-NEXT:         | | | |-PseudoObjectExpr [[ADDR_49:0x[a-z0-9]*]] <col:10, col:22> 'int'
// CXX_INT-NEXT:         | | | | |-CallExpr [[ADDR_50:0x[a-z0-9]*]] <col:10, col:22> 'int'
// CXX_INT-NEXT:         | | | | | `-ImplicitCastExpr [[ADDR_51:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX_INT-NEXT:         | | | | |   `-DeclRefExpr [[ADDR_52:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CXX_INT-NEXT:         | | | | `-CallExpr [[ADDR_53:0x[a-z0-9]*]] <line:38:1, line:67:22> 'int'
// CXX_INT-NEXT:         | | | |   `-ImplicitCastExpr [[ADDR_54:0x[a-z0-9]*]] <line:38:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX_INT-NEXT:         | | | |     `-DeclRefExpr [[ADDR_5]] <col:1> 'int ({{.*}})' Function [[ADDR_6]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int ({{.*}})'
// CXX_INT-NEXT:         | | | `-PseudoObjectExpr [[ADDR_55:0x[a-z0-9]*]] <line:67:26, col:39> 'int'
// CXX_INT-NEXT:         | | |   |-CallExpr [[ADDR_56:0x[a-z0-9]*]] <col:26, col:39> 'int'
// CXX_INT-NEXT:         | | |   | |-ImplicitCastExpr [[ADDR_57:0x[a-z0-9]*]] <col:26> 'int (*)(int)' <FunctionToPointerDecay>
// CXX_INT-NEXT:         | | |   | | `-DeclRefExpr [[ADDR_58:0x[a-z0-9]*]] <col:26> 'int (int)' {{.*}}Function [[ADDR_7]] 'also_before' 'int (int)'
// CXX_INT-NEXT:         | | |   | `-IntegerLiteral [[ADDR_59:0x[a-z0-9]*]] <col:38> 'int' 1
// CXX_INT-NEXT:         | | |   `-CallExpr [[ADDR_60:0x[a-z0-9]*]] <line:42:1, line:67:39> 'int'
// CXX_INT-NEXT:         | | |     |-ImplicitCastExpr [[ADDR_61:0x[a-z0-9]*]] <line:42:1> 'int (*)(int)' <FunctionToPointerDecay>
// CXX_INT-NEXT:         | | |     | `-DeclRefExpr [[ADDR_13]] <col:1> 'int (int)' Function [[ADDR_14]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int (int)'
// CXX_INT-NEXT:         | | |     `-IntegerLiteral [[ADDR_59]] <line:67:38> 'int' 1
// CXX_INT-NEXT:         | | `-PseudoObjectExpr [[ADDR_62:0x[a-z0-9]*]] <col:43, col:59> 'int'
// CXX_INT-NEXT:         | |   |-CallExpr [[ADDR_63:0x[a-z0-9]*]] <col:43, col:59> 'int'
// CXX_INT-NEXT:         | |   | |-ImplicitCastExpr [[ADDR_64:0x[a-z0-9]*]] <col:43> 'int (*)(int)' <FunctionToPointerDecay>
// CXX_INT-NEXT:         | |   | | `-DeclRefExpr [[ADDR_65:0x[a-z0-9]*]] <col:43> 'int (int)' {{.*}}Function [[ADDR_7]] 'also_before' 'int (int)'
// CXX_INT-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_66:0x[a-z0-9]*]] <col:55> 'int' <FloatingToIntegral>
// CXX_INT-NEXT:         | |   |   `-FloatingLiteral [[ADDR_67:0x[a-z0-9]*]] <col:55> 'float' 2.000000e+00
// CXX_INT-NEXT:         | |   `-CallExpr [[ADDR_68:0x[a-z0-9]*]] <line:42:1, line:67:59> 'int'
// CXX_INT-NEXT:         | |     |-ImplicitCastExpr [[ADDR_69:0x[a-z0-9]*]] <line:42:1> 'int (*)(int)' <FunctionToPointerDecay>
// CXX_INT-NEXT:         | |     | `-DeclRefExpr [[ADDR_13]] <col:1> 'int (int)' Function [[ADDR_14]] 'also_before[implementation={extension(disable_implicit_base)}]' 'int (int)'
// CXX_INT-NEXT:         | |     `-ImplicitCastExpr [[ADDR_70:0x[a-z0-9]*]] <line:67:55> 'int' <FloatingToIntegral>
// CXX_INT-NEXT:         | |       `-FloatingLiteral [[ADDR_67]] <col:55> 'float' 2.000000e+00
// CXX_INT-NEXT:         | `-CallExpr [[ADDR_71:0x[a-z0-9]*]] <col:63, col:77> 'int'
// CXX_INT-NEXT:         |   |-ImplicitCastExpr [[ADDR_72:0x[a-z0-9]*]] <col:63> 'int (*)(double)' <FunctionToPointerDecay>
// CXX_INT-NEXT:         |   | `-DeclRefExpr [[ADDR_73:0x[a-z0-9]*]] <col:63> 'int (double)' {{.*}}Function [[ADDR_32]] 'also_after' 'int (double)'
// CXX_INT-NEXT:         |   `-FloatingLiteral [[ADDR_74:0x[a-z0-9]*]] <col:74> 'double' 3.000000e+00
// CXX_INT-NEXT:         `-CallExpr [[ADDR_75:0x[a-z0-9]*]] <col:81, col:94> 'int'
// CXX_INT-NEXT:           |-ImplicitCastExpr [[ADDR_76:0x[a-z0-9]*]] <col:81> 'int (*)(long)' <FunctionToPointerDecay>
// CXX_INT-NEXT:           | `-DeclRefExpr [[ADDR_77:0x[a-z0-9]*]] <col:81> 'int (long)' {{.*}}Function [[ADDR_37]] 'also_after' 'int (long)'
// CXX_INT-NEXT:           `-IntegerLiteral [[ADDR_78:0x[a-z0-9]*]] <col:92> 'long' 4
