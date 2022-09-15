// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test_one(int x) {
#pragma omp distribute parallel for simd
  for (int i = 0; i < x; i++)
    ;
}

void test_two(int x, int y) {
#pragma omp distribute parallel for simd
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_three(int x, int y) {
#pragma omp distribute parallel for simd collapse(1)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_four(int x, int y) {
#pragma omp distribute parallel for simd collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_five(int x, int y, int z) {
#pragma omp distribute parallel for simd collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      for (int i = 0; i < z; i++)
        ;
}

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:3:6 test_one 'void (int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_1:0x[a-z0-9]*]] <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_2:0x[a-z0-9]*]] <col:22, line:7:1>
// CHECK-NEXT: |   `-OMPDistributeParallelForSimdDirective [[ADDR_3:0x[a-z0-9]*]] <line:4:1, col:41>
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_4:0x[a-z0-9]*]] <line:5:3, line:6:5>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_5:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-ForStmt [[ADDR_6:0x[a-z0-9]*]] <line:5:3, line:6:5>
// CHECK-NEXT: |       | | |-DeclStmt [[ADDR_7:0x[a-z0-9]*]] <line:5:8, col:17>
// CHECK-NEXT: |       | | | `-VarDecl [[ADDR_8:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   `-IntegerLiteral [[ADDR_9:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator [[ADDR_10:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr [[ADDR_11:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr [[ADDR_12:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_8]] 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr [[ADDR_13:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr [[ADDR_14:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | |-UnaryOperator [[ADDR_15:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr [[ADDR_16:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_8]] 'i' 'int'
// CHECK-NEXT: |       | | `-NullStmt [[ADDR_17:0x[a-z0-9]*]] <line:6:5>
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_18:0x[a-z0-9]*]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_19:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_20:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK-NEXT: |       | `-VarDecl [[ADDR_8]] <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   `-IntegerLiteral [[ADDR_9]] <col:16> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_21:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_22:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_23:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_24:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_25:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_26:0x[a-z0-9]*]] <line:9:1, line:14:1> line:9:6 test_two 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_27:0x[a-z0-9]*]] <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_28:0x[a-z0-9]*]] <col:22, col:26> col:26 used y 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_29:0x[a-z0-9]*]] <col:29, line:14:1>
// CHECK-NEXT: |   `-OMPDistributeParallelForSimdDirective [[ADDR_30:0x[a-z0-9]*]] <line:10:1, col:41>
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_31:0x[a-z0-9]*]] <line:11:3, line:13:7>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_32:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-ForStmt [[ADDR_33:0x[a-z0-9]*]] <line:11:3, line:13:7>
// CHECK-NEXT: |       | | |-DeclStmt [[ADDR_34:0x[a-z0-9]*]] <line:11:8, col:17>
// CHECK-NEXT: |       | | | `-VarDecl [[ADDR_35:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   `-IntegerLiteral [[ADDR_36:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator [[ADDR_37:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr [[ADDR_38:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr [[ADDR_39:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_35]] 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr [[ADDR_40:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr [[ADDR_41:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_27]] 'x' 'int'
// CHECK-NEXT: |       | | |-UnaryOperator [[ADDR_42:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr [[ADDR_43:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_35]] 'i' 'int'
// CHECK-NEXT: |       | | `-ForStmt [[ADDR_44:0x[a-z0-9]*]] <line:12:5, line:13:7>
// CHECK-NEXT: |       | |   |-DeclStmt [[ADDR_45:0x[a-z0-9]*]] <line:12:10, col:19>
// CHECK-NEXT: |       | |   | `-VarDecl [[ADDR_46:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | |   |   `-IntegerLiteral [[ADDR_47:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT: |       | |   |-<<<NULL>>>
// CHECK-NEXT: |       | |   |-BinaryOperator [[ADDR_48:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | |   | |-ImplicitCastExpr [[ADDR_49:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   | | `-DeclRefExpr [[ADDR_50:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_46]] 'i' 'int'
// CHECK-NEXT: |       | |   | `-ImplicitCastExpr [[ADDR_51:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   |   `-DeclRefExpr [[ADDR_52:0x[a-z0-9]*]] <col:25> 'int' {{.*}}ParmVar [[ADDR_28]] 'y' 'int'
// CHECK-NEXT: |       | |   |-UnaryOperator [[ADDR_53:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | |   | `-DeclRefExpr [[ADDR_54:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_46]] 'i' 'int'
// CHECK-NEXT: |       | |   `-NullStmt [[ADDR_55:0x[a-z0-9]*]] <line:13:7>
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_56:0x[a-z0-9]*]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_57:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_58:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-distribute-parallel-for-simd.c:10:1) *const restrict'
// CHECK-NEXT: |       | |-VarDecl [[ADDR_35]] <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | `-IntegerLiteral [[ADDR_36]] <col:16> 'int' 0
// CHECK-NEXT: |       | `-VarDecl [[ADDR_46]] <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   `-IntegerLiteral [[ADDR_47]] <col:18> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_59:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_60:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_61:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_62:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_63:0x[a-z0-9]*]] <line:11:23> 'int' {{.*}}ParmVar [[ADDR_27]] 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_64:0x[a-z0-9]*]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_28]] 'y' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_65:0x[a-z0-9]*]] <line:16:1, line:21:1> line:16:6 test_three 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_66:0x[a-z0-9]*]] <col:17, col:21> col:21 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_67:0x[a-z0-9]*]] <col:24, col:28> col:28 used y 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_68:0x[a-z0-9]*]] <col:31, line:21:1>
// CHECK-NEXT: |   `-OMPDistributeParallelForSimdDirective [[ADDR_69:0x[a-z0-9]*]] <line:17:1, col:53>
// CHECK-NEXT: |     |-OMPCollapseClause [[ADDR_70:0x[a-z0-9]*]] <col:42, col:52>
// CHECK-NEXT: |     | `-ConstantExpr [[ADDR_71:0x[a-z0-9]*]] <col:51> 'int'
// CHECK-NEXT: |     |   |-value: Int 1
// CHECK-NEXT: |     |   `-IntegerLiteral [[ADDR_72:0x[a-z0-9]*]] <col:51> 'int' 1
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_73:0x[a-z0-9]*]] <line:18:3, line:20:7>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_74:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-ForStmt [[ADDR_75:0x[a-z0-9]*]] <line:18:3, line:20:7>
// CHECK-NEXT: |       | | |-DeclStmt [[ADDR_76:0x[a-z0-9]*]] <line:18:8, col:17>
// CHECK-NEXT: |       | | | `-VarDecl [[ADDR_77:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   `-IntegerLiteral [[ADDR_78:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator [[ADDR_79:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr [[ADDR_80:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr [[ADDR_81:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_77]] 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr [[ADDR_82:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr [[ADDR_83:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_66]] 'x' 'int'
// CHECK-NEXT: |       | | |-UnaryOperator [[ADDR_84:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr [[ADDR_85:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_77]] 'i' 'int'
// CHECK-NEXT: |       | | `-ForStmt [[ADDR_86:0x[a-z0-9]*]] <line:19:5, line:20:7>
// CHECK-NEXT: |       | |   |-DeclStmt [[ADDR_87:0x[a-z0-9]*]] <line:19:10, col:19>
// CHECK-NEXT: |       | |   | `-VarDecl [[ADDR_88:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | |   |   `-IntegerLiteral [[ADDR_89:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT: |       | |   |-<<<NULL>>>
// CHECK-NEXT: |       | |   |-BinaryOperator [[ADDR_90:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | |   | |-ImplicitCastExpr [[ADDR_91:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   | | `-DeclRefExpr [[ADDR_92:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_88]] 'i' 'int'
// CHECK-NEXT: |       | |   | `-ImplicitCastExpr [[ADDR_93:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   |   `-DeclRefExpr [[ADDR_94:0x[a-z0-9]*]] <col:25> 'int' {{.*}}ParmVar [[ADDR_67]] 'y' 'int'
// CHECK-NEXT: |       | |   |-UnaryOperator [[ADDR_95:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | |   | `-DeclRefExpr [[ADDR_96:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_88]] 'i' 'int'
// CHECK-NEXT: |       | |   `-NullStmt [[ADDR_97:0x[a-z0-9]*]] <line:20:7>
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_98:0x[a-z0-9]*]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_99:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_100:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-distribute-parallel-for-simd.c:17:1) *const restrict'
// CHECK-NEXT: |       | |-VarDecl [[ADDR_77]] <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | `-IntegerLiteral [[ADDR_78]] <col:16> 'int' 0
// CHECK-NEXT: |       | `-VarDecl [[ADDR_88]] <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   `-IntegerLiteral [[ADDR_89]] <col:18> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_101:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_102:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_103:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_104:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_105:0x[a-z0-9]*]] <line:18:23> 'int' {{.*}}ParmVar [[ADDR_66]] 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_106:0x[a-z0-9]*]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_67]] 'y' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_107:0x[a-z0-9]*]] <line:23:1, line:28:1> line:23:6 test_four 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_108:0x[a-z0-9]*]] <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_109:0x[a-z0-9]*]] <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_110:0x[a-z0-9]*]] <col:30, line:28:1>
// CHECK-NEXT: |   `-OMPDistributeParallelForSimdDirective [[ADDR_111:0x[a-z0-9]*]] <line:24:1, col:53>
// CHECK-NEXT: |     |-OMPCollapseClause [[ADDR_112:0x[a-z0-9]*]] <col:42, col:52>
// CHECK-NEXT: |     | `-ConstantExpr [[ADDR_113:0x[a-z0-9]*]] <col:51> 'int'
// CHECK-NEXT: |     |   |-value: Int 2
// CHECK-NEXT: |     |   `-IntegerLiteral [[ADDR_114:0x[a-z0-9]*]] <col:51> 'int' 2
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_115:0x[a-z0-9]*]] <line:25:3, line:27:7>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_116:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-ForStmt [[ADDR_117:0x[a-z0-9]*]] <line:25:3, line:27:7>
// CHECK-NEXT: |       | | |-DeclStmt [[ADDR_118:0x[a-z0-9]*]] <line:25:8, col:17>
// CHECK-NEXT: |       | | | `-VarDecl [[ADDR_119:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   `-IntegerLiteral [[ADDR_120:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator [[ADDR_121:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr [[ADDR_122:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr [[ADDR_123:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_119]] 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr [[ADDR_124:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr [[ADDR_125:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_108]] 'x' 'int'
// CHECK-NEXT: |       | | |-UnaryOperator [[ADDR_126:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr [[ADDR_127:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_119]] 'i' 'int'
// CHECK-NEXT: |       | | `-ForStmt [[ADDR_128:0x[a-z0-9]*]] <line:26:5, line:27:7>
// CHECK-NEXT: |       | |   |-DeclStmt [[ADDR_129:0x[a-z0-9]*]] <line:26:10, col:19>
// CHECK-NEXT: |       | |   | `-VarDecl [[ADDR_130:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | |   |   `-IntegerLiteral [[ADDR_131:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT: |       | |   |-<<<NULL>>>
// CHECK-NEXT: |       | |   |-BinaryOperator [[ADDR_132:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | |   | |-ImplicitCastExpr [[ADDR_133:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   | | `-DeclRefExpr [[ADDR_134:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_130]] 'i' 'int'
// CHECK-NEXT: |       | |   | `-ImplicitCastExpr [[ADDR_135:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   |   `-DeclRefExpr [[ADDR_136:0x[a-z0-9]*]] <col:25> 'int' {{.*}}ParmVar [[ADDR_109]] 'y' 'int'
// CHECK-NEXT: |       | |   |-UnaryOperator [[ADDR_137:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | |   | `-DeclRefExpr [[ADDR_138:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_130]] 'i' 'int'
// CHECK-NEXT: |       | |   `-NullStmt [[ADDR_139:0x[a-z0-9]*]] <line:27:7>
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_140:0x[a-z0-9]*]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_141:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_142:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-distribute-parallel-for-simd.c:24:1) *const restrict'
// CHECK-NEXT: |       | |-VarDecl [[ADDR_119]] <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | `-IntegerLiteral [[ADDR_120]] <col:16> 'int' 0
// CHECK-NEXT: |       | `-VarDecl [[ADDR_130]] <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   `-IntegerLiteral [[ADDR_131]] <col:18> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_143:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_144:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_145:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_146:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_147:0x[a-z0-9]*]] <line:25:23> 'int' {{.*}}ParmVar [[ADDR_108]] 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_148:0x[a-z0-9]*]] <line:26:25> 'int' {{.*}}ParmVar [[ADDR_109]] 'y' 'int'
// CHECK-NEXT: `-FunctionDecl [[ADDR_149:0x[a-z0-9]*]] <line:30:1, line:36:1> line:30:6 test_five 'void (int, int, int)'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_150:0x[a-z0-9]*]] <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_151:0x[a-z0-9]*]] <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_152:0x[a-z0-9]*]] <col:30, col:34> col:34 used z 'int'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_153:0x[a-z0-9]*]] <col:37, line:36:1>
// CHECK-NEXT:     `-OMPDistributeParallelForSimdDirective [[ADDR_154:0x[a-z0-9]*]] <line:31:1, col:53>
// CHECK-NEXT:       |-OMPCollapseClause [[ADDR_155:0x[a-z0-9]*]] <col:42, col:52>
// CHECK-NEXT:       | `-ConstantExpr [[ADDR_156:0x[a-z0-9]*]] <col:51> 'int'
// CHECK-NEXT:       |   |-value: Int 2
// CHECK-NEXT:       |   `-IntegerLiteral [[ADDR_157:0x[a-z0-9]*]] <col:51> 'int' 2
// CHECK-NEXT:       `-CapturedStmt [[ADDR_158:0x[a-z0-9]*]] <line:32:3, line:35:9>
// CHECK-NEXT:         |-CapturedDecl [[ADDR_159:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | |-ForStmt [[ADDR_160:0x[a-z0-9]*]] <line:32:3, line:35:9>
// CHECK-NEXT:         | | |-DeclStmt [[ADDR_161:0x[a-z0-9]*]] <line:32:8, col:17>
// CHECK-NEXT:         | | | `-VarDecl [[ADDR_162:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | |   `-IntegerLiteral [[ADDR_163:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT:         | | |-<<<NULL>>>
// CHECK-NEXT:         | | |-BinaryOperator [[ADDR_164:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | |-ImplicitCastExpr [[ADDR_165:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | `-DeclRefExpr [[ADDR_166:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_162]] 'i' 'int'
// CHECK-NEXT:         | | | `-ImplicitCastExpr [[ADDR_167:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   `-DeclRefExpr [[ADDR_168:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_150]] 'x' 'int'
// CHECK-NEXT:         | | |-UnaryOperator [[ADDR_169:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | `-DeclRefExpr [[ADDR_170:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_162]] 'i' 'int'
// CHECK-NEXT:         | | `-ForStmt [[ADDR_171:0x[a-z0-9]*]] <line:33:5, line:35:9>
// CHECK-NEXT:         | |   |-DeclStmt [[ADDR_172:0x[a-z0-9]*]] <line:33:10, col:19>
// CHECK-NEXT:         | |   | `-VarDecl [[ADDR_173:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | |   |   `-IntegerLiteral [[ADDR_174:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT:         | |   |-<<<NULL>>>
// CHECK-NEXT:         | |   |-BinaryOperator [[ADDR_175:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | |   | |-ImplicitCastExpr [[ADDR_176:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | |   | | `-DeclRefExpr [[ADDR_177:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_173]] 'i' 'int'
// CHECK-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_178:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | |   |   `-DeclRefExpr [[ADDR_179:0x[a-z0-9]*]] <col:25> 'int' {{.*}}ParmVar [[ADDR_151]] 'y' 'int'
// CHECK-NEXT:         | |   |-UnaryOperator [[ADDR_180:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | |   | `-DeclRefExpr [[ADDR_181:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_173]] 'i' 'int'
// CHECK-NEXT:         | |   `-ForStmt [[ADDR_182:0x[a-z0-9]*]] <line:34:7, line:35:9>
// CHECK-NEXT:         | |     |-DeclStmt [[ADDR_183:0x[a-z0-9]*]] <line:34:12, col:21>
// CHECK-NEXT:         | |     | `-VarDecl [[ADDR_184:0x[a-z0-9]*]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | |     |   `-IntegerLiteral [[ADDR_185:0x[a-z0-9]*]] <col:20> 'int' 0
// CHECK-NEXT:         | |     |-<<<NULL>>>
// CHECK-NEXT:         | |     |-BinaryOperator [[ADDR_186:0x[a-z0-9]*]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | |     | |-ImplicitCastExpr [[ADDR_187:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | |     | | `-DeclRefExpr [[ADDR_188:0x[a-z0-9]*]] <col:23> 'int' {{.*}}Var [[ADDR_184]] 'i' 'int'
// CHECK-NEXT:         | |     | `-ImplicitCastExpr [[ADDR_189:0x[a-z0-9]*]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | |     |   `-DeclRefExpr [[ADDR_190:0x[a-z0-9]*]] <col:27> 'int' {{.*}}ParmVar [[ADDR_152]] 'z' 'int'
// CHECK-NEXT:         | |     |-UnaryOperator [[ADDR_191:0x[a-z0-9]*]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | |     | `-DeclRefExpr [[ADDR_192:0x[a-z0-9]*]] <col:30> 'int' {{.*}}Var [[ADDR_184]] 'i' 'int'
// CHECK-NEXT:         | |     `-NullStmt [[ADDR_193:0x[a-z0-9]*]] <line:35:9>
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_194:0x[a-z0-9]*]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_195:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_196:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-distribute-parallel-for-simd.c:31:1) *const restrict'
// CHECK-NEXT:         | |-VarDecl [[ADDR_162]] <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | `-IntegerLiteral [[ADDR_163]] <col:16> 'int' 0
// CHECK-NEXT:         | |-VarDecl [[ADDR_173]] <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | `-IntegerLiteral [[ADDR_174]] <col:18> 'int' 0
// CHECK-NEXT:         | `-VarDecl [[ADDR_184]] <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   `-IntegerLiteral [[ADDR_185]] <col:20> 'int' 0
// CHECK-NEXT:         |-DeclRefExpr [[ADDR_197:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_198:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT:         |-DeclRefExpr [[ADDR_199:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_200:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT:         |-DeclRefExpr [[ADDR_201:0x[a-z0-9]*]] <line:32:23> 'int' {{.*}}ParmVar [[ADDR_150]] 'x' 'int'
// CHECK-NEXT:         |-DeclRefExpr [[ADDR_202:0x[a-z0-9]*]] <line:33:25> 'int' {{.*}}ParmVar [[ADDR_151]] 'y' 'int'
// CHECK-NEXT:         `-DeclRefExpr [[ADDR_203:0x[a-z0-9]*]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_152]] 'z' 'int'
