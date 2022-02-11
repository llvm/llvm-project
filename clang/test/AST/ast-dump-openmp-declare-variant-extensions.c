// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

int picked1(void) { return 0; }
int picked2(void) { return 0; }
int picked3(void);
int picked4(void);
int picked5(void) { return 0; }
int picked6(void) { return 0; }
int picked7(void) { return 0; }
int not_picked1(void) { return 1; }
int not_picked2(void) { return 2; }
int not_picked3(void);
int not_picked4(void);
int not_picked5(void);
int not_picked6(void);

#pragma omp declare variant(picked1) match(implementation={extension(match_any)}, device={kind(cpu, gpu)})
int base1(void) { return 3; }

#pragma omp declare variant(picked2) match(implementation={extension(match_none)}, device={kind(gpu, fpga)})
int base2(void) { return 4; }

#pragma omp declare variant(picked3) match(implementation={vendor(pgi), extension(match_any)}, device={kind(cpu, gpu)})
int base3(void) { return 5; }

#pragma omp declare variant(picked4) match(user={condition(0)}, implementation={extension(match_none)}, device={kind(gpu, fpga)})
int base4(void) { return 6; }

#pragma omp declare variant(picked5) match(user={condition(1)}, implementation={extension(match_all)}, device={kind(cpu)})
int base5(void) { return 7; }

#pragma omp declare variant(not_picked1) match(implementation={extension(match_any)}, device={kind(gpu, fpga)})
int base6(void) { return 0; }

#pragma omp declare variant(not_picked2) match(implementation={extension(match_none)}, device={kind(gpu, cpu)})
int base7(void) { return 0; }

#pragma omp declare variant(not_picked3) match(implementation={vendor(llvm), extension(match_any)}, device={kind(fpga, gpu)})
int base8(void) { return 0; }

#pragma omp declare variant(not_picked4) match(user={condition(1)}, implementation={extension(match_none)}, device={kind(gpu, fpga)})
int base9(void) { return 0; }

#pragma omp declare variant(not_picked5) match(user={condition(1)}, implementation={extension(match_all)}, device={kind(cpu, gpu)})
int base10(void) { return 0; }

#pragma omp declare variant(not_picked6) match(implementation={extension(match_any)})
int base11(void) { return 0; }

#pragma omp declare variant(picked6) match(implementation={extension(match_all)})
int base12(void) { return 8; }

#pragma omp declare variant(picked7) match(implementation={extension(match_none)})
int base13(void) { return 9; }

#pragma omp begin declare variant match(implementation={extension(match_any)}, device={kind(cpu, gpu)})
int overloaded1(void) { return 0; }
#pragma omp end declare variant

int overloaded2(void) { return 1; }
#pragma omp begin declare variant match(implementation={extension(match_none)}, device={kind(fpga, gpu)})
int overloaded2(void) { return 0; }
#pragma omp end declare variant

#pragma omp begin declare variant match(implementation={extension(match_none)}, device={kind(cpu)})
NOT PARSED
#pragma omp end declare variant


int picked3(void) { return 0; }
int picked4(void) { return 0; }
int not_picked3(void) { return 10; }
int not_picked4(void) { return 11; }
int not_picked5(void) { return 12; }
int not_picked6(void) { return 13; }

int test(void) {
  // Should return 0.
  return base1() + base2() + base3() + base4() + base5() + base6() + base7() +
         base8() + base9() + base10() + base11() + base12() + base13() +
         overloaded1() + overloaded2();
}

// 1) All "picked" versions are called but none of the "non_picked" ones is.
// 2) The overloaded functions that return 0 are called.

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, col:31> col:5 referenced picked1 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:19, col:31>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <col:21, col:28>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:28> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_4:0x[a-z0-9]*]] <line:6:1, col:31> col:5 referenced picked2 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_5:0x[a-z0-9]*]] <col:19, col:31>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_6:0x[a-z0-9]*]] <col:21, col:28>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_7:0x[a-z0-9]*]] <col:28> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_8:0x[a-z0-9]*]] <line:7:1, col:17> col:5 referenced picked3 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_9:0x[a-z0-9]*]] <line:8:1, col:17> col:5 referenced picked4 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_10:0x[a-z0-9]*]] <line:9:1, col:31> col:5 referenced picked5 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:19, col:31>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <col:21, col:28>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:28> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_14:0x[a-z0-9]*]] <line:10:1, col:31> col:5 referenced picked6 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_15:0x[a-z0-9]*]] <col:19, col:31>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_16:0x[a-z0-9]*]] <col:21, col:28>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_17:0x[a-z0-9]*]] <col:28> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_18:0x[a-z0-9]*]] <line:11:1, col:31> col:5 referenced picked7 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_19:0x[a-z0-9]*]] <col:19, col:31>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_20:0x[a-z0-9]*]] <col:21, col:28>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_21:0x[a-z0-9]*]] <col:28> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:12:1, col:35> col:5 referenced not_picked1 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_23:0x[a-z0-9]*]] <col:23, col:35>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_24:0x[a-z0-9]*]] <col:25, col:32>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_25:0x[a-z0-9]*]] <col:32> 'int' 1
// CHECK-NEXT: |-FunctionDecl [[ADDR_26:0x[a-z0-9]*]] <line:13:1, col:35> col:5 referenced not_picked2 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_27:0x[a-z0-9]*]] <col:23, col:35>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_28:0x[a-z0-9]*]] <col:25, col:32>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_29:0x[a-z0-9]*]] <col:32> 'int' 2
// CHECK-NEXT: |-FunctionDecl [[ADDR_30:0x[a-z0-9]*]] <line:14:1, col:21> col:5 referenced not_picked3 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_31:0x[a-z0-9]*]] <line:15:1, col:21> col:5 referenced not_picked4 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_32:0x[a-z0-9]*]] <line:16:1, col:21> col:5 referenced not_picked5 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_33:0x[a-z0-9]*]] <line:17:1, col:21> col:5 referenced not_picked6 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_34:0x[a-z0-9]*]] <line:20:1, col:29> col:5 used base1 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_35:0x[a-z0-9]*]] <col:17, col:29>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_36:0x[a-z0-9]*]] <col:19, col:26>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_37:0x[a-z0-9]*]] <col:26> 'int' 3
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_38:0x[a-z0-9]*]] <line:19:1, col:107> Implicit implementation={extension(match_any)}, device={kind(cpu, gpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_39:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'picked1' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_40:0x[a-z0-9]*]] <line:23:1, col:29> col:5 used base2 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_41:0x[a-z0-9]*]] <col:17, col:29>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_42:0x[a-z0-9]*]] <col:19, col:26>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_43:0x[a-z0-9]*]] <col:26> 'int' 4
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_44:0x[a-z0-9]*]] <line:22:1, col:109> Implicit implementation={extension(match_none)}, device={kind(gpu, fpga)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_45:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_4]] 'picked2' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_46:0x[a-z0-9]*]] <line:26:1, col:29> col:5 used base3 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_47:0x[a-z0-9]*]] <col:17, col:29>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_48:0x[a-z0-9]*]] <col:19, col:26>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_49:0x[a-z0-9]*]] <col:26> 'int' 5
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_50:0x[a-z0-9]*]] <line:25:1, col:120> Implicit implementation={vendor(pgi), extension(match_any)}, device={kind(cpu, gpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_51:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_8]] 'picked3' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_52:0x[a-z0-9]*]] <line:29:1, col:29> col:5 used base4 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_53:0x[a-z0-9]*]] <col:17, col:29>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_54:0x[a-z0-9]*]] <col:19, col:26>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_55:0x[a-z0-9]*]] <col:26> 'int' 6
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_56:0x[a-z0-9]*]] <line:28:1, col:130> Implicit user={condition(0)}, implementation={extension(match_none)}, device={kind(gpu, fpga)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_57:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_9]] 'picked4' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_58:0x[a-z0-9]*]] <line:32:1, col:29> col:5 used base5 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_59:0x[a-z0-9]*]] <col:17, col:29>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_60:0x[a-z0-9]*]] <col:19, col:26>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_61:0x[a-z0-9]*]] <col:26> 'int' 7
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_62:0x[a-z0-9]*]] <line:31:1, col:123> Implicit user={condition(1)}, implementation={extension(match_all)}, device={kind(cpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_63:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_10]] 'picked5' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_64:0x[a-z0-9]*]] <line:35:1, col:29> col:5 used base6 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_65:0x[a-z0-9]*]] <col:17, col:29>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_66:0x[a-z0-9]*]] <col:19, col:26>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_67:0x[a-z0-9]*]] <col:26> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_68:0x[a-z0-9]*]] <line:34:1, col:112> Implicit implementation={extension(match_any)}, device={kind(gpu, fpga)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_69:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_22]] 'not_picked1' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_70:0x[a-z0-9]*]] <line:38:1, col:29> col:5 used base7 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_71:0x[a-z0-9]*]] <col:17, col:29>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_72:0x[a-z0-9]*]] <col:19, col:26>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_73:0x[a-z0-9]*]] <col:26> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_74:0x[a-z0-9]*]] <line:37:1, col:112> Implicit implementation={extension(match_none)}, device={kind(gpu, cpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_75:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_26]] 'not_picked2' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_76:0x[a-z0-9]*]] <line:41:1, col:29> col:5 used base8 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_77:0x[a-z0-9]*]] <col:17, col:29>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_78:0x[a-z0-9]*]] <col:19, col:26>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_79:0x[a-z0-9]*]] <col:26> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_80:0x[a-z0-9]*]] <line:40:1, col:126> Implicit implementation={vendor(llvm), extension(match_any)}, device={kind(fpga, gpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_81:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_30]] 'not_picked3' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_82:0x[a-z0-9]*]] <line:44:1, col:29> col:5 used base9 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_83:0x[a-z0-9]*]] <col:17, col:29>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_84:0x[a-z0-9]*]] <col:19, col:26>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_85:0x[a-z0-9]*]] <col:26> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_86:0x[a-z0-9]*]] <line:43:1, col:134> Implicit user={condition(1)}, implementation={extension(match_none)}, device={kind(gpu, fpga)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_87:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_31]] 'not_picked4' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_88:0x[a-z0-9]*]] <line:47:1, col:30> col:5 used base10 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_89:0x[a-z0-9]*]] <col:18, col:30>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_90:0x[a-z0-9]*]] <col:20, col:27>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_91:0x[a-z0-9]*]] <col:27> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_92:0x[a-z0-9]*]] <line:46:1, col:132> Implicit user={condition(1)}, implementation={extension(match_all)}, device={kind(cpu, gpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_93:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_32]] 'not_picked5' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_94:0x[a-z0-9]*]] <line:50:1, col:30> col:5 used base11 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_95:0x[a-z0-9]*]] <col:18, col:30>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_96:0x[a-z0-9]*]] <col:20, col:27>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_97:0x[a-z0-9]*]] <col:27> 'int' 0
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_98:0x[a-z0-9]*]] <line:49:1, col:86> Implicit implementation={extension(match_any)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_99:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_33]] 'not_picked6' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_100:0x[a-z0-9]*]] <line:53:1, col:30> col:5 used base12 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_101:0x[a-z0-9]*]] <col:18, col:30>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_102:0x[a-z0-9]*]] <col:20, col:27>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_103:0x[a-z0-9]*]] <col:27> 'int' 8
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_104:0x[a-z0-9]*]] <line:52:1, col:82> Implicit implementation={extension(match_all)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_105:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_14]] 'picked6' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_106:0x[a-z0-9]*]] <line:56:1, col:30> col:5 used base13 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_107:0x[a-z0-9]*]] <col:18, col:30>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_108:0x[a-z0-9]*]] <col:20, col:27>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_109:0x[a-z0-9]*]] <col:27> 'int' 9
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_110:0x[a-z0-9]*]] <line:55:1, col:83> Implicit implementation={extension(match_none)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_111:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_18]] 'picked7' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: |-FunctionDecl [[ADDR_112:0x[a-z0-9]*]] <line:59:1, col:21> col:5 implicit used overloaded1 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_113:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(match_any)}, device={kind(cpu, gpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_114:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_115:0x[a-z0-9]*]] 'overloaded1[implementation={extension(match_any)}, device={kind(cpu, gpu)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_115]] <col:1, col:35> col:1 overloaded1[implementation={extension(match_any)}, device={kind(cpu, gpu)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_116:0x[a-z0-9]*]] <col:23, col:35>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_117:0x[a-z0-9]*]] <col:25, col:32>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_118:0x[a-z0-9]*]] <col:32> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_119:0x[a-z0-9]*]] <line:62:1, col:35> col:5 used overloaded2 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_120:0x[a-z0-9]*]] <col:23, col:35>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_121:0x[a-z0-9]*]] <col:25, col:32>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_122:0x[a-z0-9]*]] <col:32> 'int' 1
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_123:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(match_none)}, device={kind(fpga, gpu)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_124:0x[a-z0-9]*]] <line:64:1> 'int ({{.*}})' {{.*}}Function [[ADDR_125:0x[a-z0-9]*]] 'overloaded2[implementation={extension(match_none)}, device={kind(fpga, gpu)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_125]] <col:1, col:35> col:1 overloaded2[implementation={extension(match_none)}, device={kind(fpga, gpu)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_126:0x[a-z0-9]*]] <col:23, col:35>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_127:0x[a-z0-9]*]] <col:25, col:32>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_128:0x[a-z0-9]*]] <col:32> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_129:0x[a-z0-9]*]] prev [[ADDR_8]] <line:72:1, col:31> col:5 picked3 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_130:0x[a-z0-9]*]] <col:19, col:31>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_131:0x[a-z0-9]*]] <col:21, col:28>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_132:0x[a-z0-9]*]] <col:28> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_133:0x[a-z0-9]*]] prev [[ADDR_9]] <line:73:1, col:31> col:5 picked4 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_134:0x[a-z0-9]*]] <col:19, col:31>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_135:0x[a-z0-9]*]] <col:21, col:28>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_136:0x[a-z0-9]*]] <col:28> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_137:0x[a-z0-9]*]] prev [[ADDR_30]] <line:74:1, col:36> col:5 not_picked3 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_138:0x[a-z0-9]*]] <col:23, col:36>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_139:0x[a-z0-9]*]] <col:25, col:32>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_140:0x[a-z0-9]*]] <col:32> 'int' 10
// CHECK-NEXT: |-FunctionDecl [[ADDR_141:0x[a-z0-9]*]] prev [[ADDR_31]] <line:75:1, col:36> col:5 not_picked4 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_142:0x[a-z0-9]*]] <col:23, col:36>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_143:0x[a-z0-9]*]] <col:25, col:32>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_144:0x[a-z0-9]*]] <col:32> 'int' 11
// CHECK-NEXT: |-FunctionDecl [[ADDR_145:0x[a-z0-9]*]] prev [[ADDR_32]] <line:76:1, col:36> col:5 not_picked5 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_146:0x[a-z0-9]*]] <col:23, col:36>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_147:0x[a-z0-9]*]] <col:25, col:32>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_148:0x[a-z0-9]*]] <col:32> 'int' 12
// CHECK-NEXT: |-FunctionDecl [[ADDR_149:0x[a-z0-9]*]] prev [[ADDR_33]] <line:77:1, col:36> col:5 not_picked6 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_150:0x[a-z0-9]*]] <col:23, col:36>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_151:0x[a-z0-9]*]] <col:25, col:32>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_152:0x[a-z0-9]*]] <col:32> 'int' 13
// CHECK-NEXT: `-FunctionDecl [[ADDR_153:0x[a-z0-9]*]] <line:79:1, line:84:1> line:79:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_154:0x[a-z0-9]*]] <col:16, line:84:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_155:0x[a-z0-9]*]] <line:81:3, line:83:38>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_156:0x[a-z0-9]*]] <line:81:10, line:83:38> 'int' '+'
// CHECK-NEXT:         |-BinaryOperator [[ADDR_157:0x[a-z0-9]*]] <line:81:10, line:83:22> 'int' '+'
// CHECK-NEXT:         | |-BinaryOperator [[ADDR_158:0x[a-z0-9]*]] <line:81:10, line:82:70> 'int' '+'
// CHECK-NEXT:         | | |-BinaryOperator [[ADDR_159:0x[a-z0-9]*]] <line:81:10, line:82:59> 'int' '+'
// CHECK-NEXT:         | | | |-BinaryOperator [[ADDR_160:0x[a-z0-9]*]] <line:81:10, line:82:48> 'int' '+'
// CHECK-NEXT:         | | | | |-BinaryOperator [[ADDR_161:0x[a-z0-9]*]] <line:81:10, line:82:37> 'int' '+'
// CHECK-NEXT:         | | | | | |-BinaryOperator [[ADDR_162:0x[a-z0-9]*]] <line:81:10, line:82:26> 'int' '+'
// CHECK-NEXT:         | | | | | | |-BinaryOperator [[ADDR_163:0x[a-z0-9]*]] <line:81:10, line:82:16> 'int' '+'
// CHECK-NEXT:         | | | | | | | |-BinaryOperator [[ADDR_164:0x[a-z0-9]*]] <line:81:10, col:76> 'int' '+'
// CHECK-NEXT:         | | | | | | | | |-BinaryOperator [[ADDR_165:0x[a-z0-9]*]] <col:10, col:66> 'int' '+'
// CHECK-NEXT:         | | | | | | | | | |-BinaryOperator [[ADDR_166:0x[a-z0-9]*]] <col:10, col:56> 'int' '+'
// CHECK-NEXT:         | | | | | | | | | | |-BinaryOperator [[ADDR_167:0x[a-z0-9]*]] <col:10, col:46> 'int' '+'
// CHECK-NEXT:         | | | | | | | | | | | |-BinaryOperator [[ADDR_168:0x[a-z0-9]*]] <col:10, col:36> 'int' '+'
// CHECK-NEXT:         | | | | | | | | | | | | |-BinaryOperator [[ADDR_169:0x[a-z0-9]*]] <col:10, col:26> 'int' '+'
// CHECK-NEXT:         | | | | | | | | | | | | | |-PseudoObjectExpr [[ADDR_170:0x[a-z0-9]*]] <col:10, col:16> 'int'
// CHECK-NEXT:         | | | | | | | | | | | | | | |-CallExpr [[ADDR_171:0x[a-z0-9]*]] <col:10, col:16> 'int'
// CHECK-NEXT:         | | | | | | | | | | | | | | | `-ImplicitCastExpr [[ADDR_172:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | | | | | | |   `-DeclRefExpr [[ADDR_173:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_34]] 'base1' 'int ({{.*}})'
// CHECK-NEXT:         | | | | | | | | | | | | | | `-CallExpr [[ADDR_174:0x[a-z0-9]*]] <line:19:29, line:81:16> 'int'
// CHECK-NEXT:         | | | | | | | | | | | | | |   `-ImplicitCastExpr [[ADDR_175:0x[a-z0-9]*]] <line:19:29> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | | | | | |     `-DeclRefExpr [[ADDR_39]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'picked1' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT:         | | | | | | | | | | | | | `-PseudoObjectExpr [[ADDR_176:0x[a-z0-9]*]] <line:81:20, col:26> 'int'
// CHECK-NEXT:         | | | | | | | | | | | | |   |-CallExpr [[ADDR_177:0x[a-z0-9]*]] <col:20, col:26> 'int'
// CHECK-NEXT:         | | | | | | | | | | | | |   | `-ImplicitCastExpr [[ADDR_178:0x[a-z0-9]*]] <col:20> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | | | | |   |   `-DeclRefExpr [[ADDR_179:0x[a-z0-9]*]] <col:20> 'int ({{.*}})' {{.*}}Function [[ADDR_40]] 'base2' 'int ({{.*}})'
// CHECK-NEXT:         | | | | | | | | | | | | |   `-CallExpr [[ADDR_180:0x[a-z0-9]*]] <line:22:29, line:81:26> 'int'
// CHECK-NEXT:         | | | | | | | | | | | | |     `-ImplicitCastExpr [[ADDR_181:0x[a-z0-9]*]] <line:22:29> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | | | | |       `-DeclRefExpr [[ADDR_45]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_4]] 'picked2' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT:         | | | | | | | | | | | | `-PseudoObjectExpr [[ADDR_182:0x[a-z0-9]*]] <line:81:30, col:36> 'int'
// CHECK-NEXT:         | | | | | | | | | | | |   |-CallExpr [[ADDR_183:0x[a-z0-9]*]] <col:30, col:36> 'int'
// CHECK-NEXT:         | | | | | | | | | | | |   | `-ImplicitCastExpr [[ADDR_184:0x[a-z0-9]*]] <col:30> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | | | |   |   `-DeclRefExpr [[ADDR_185:0x[a-z0-9]*]] <col:30> 'int ({{.*}})' {{.*}}Function [[ADDR_46]] 'base3' 'int ({{.*}})'
// CHECK-NEXT:         | | | | | | | | | | | |   `-CallExpr [[ADDR_186:0x[a-z0-9]*]] <line:25:29, line:81:36> 'int'
// CHECK-NEXT:         | | | | | | | | | | | |     `-ImplicitCastExpr [[ADDR_187:0x[a-z0-9]*]] <line:25:29> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | | | |       `-DeclRefExpr [[ADDR_51]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_8]] 'picked3' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT:         | | | | | | | | | | | `-PseudoObjectExpr [[ADDR_188:0x[a-z0-9]*]] <line:81:40, col:46> 'int'
// CHECK-NEXT:         | | | | | | | | | | |   |-CallExpr [[ADDR_189:0x[a-z0-9]*]] <col:40, col:46> 'int'
// CHECK-NEXT:         | | | | | | | | | | |   | `-ImplicitCastExpr [[ADDR_190:0x[a-z0-9]*]] <col:40> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | | |   |   `-DeclRefExpr [[ADDR_191:0x[a-z0-9]*]] <col:40> 'int ({{.*}})' {{.*}}Function [[ADDR_52]] 'base4' 'int ({{.*}})'
// CHECK-NEXT:         | | | | | | | | | | |   `-CallExpr [[ADDR_192:0x[a-z0-9]*]] <line:28:29, line:81:46> 'int'
// CHECK-NEXT:         | | | | | | | | | | |     `-ImplicitCastExpr [[ADDR_193:0x[a-z0-9]*]] <line:28:29> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | | |       `-DeclRefExpr [[ADDR_57]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_9]] 'picked4' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT:         | | | | | | | | | | `-PseudoObjectExpr [[ADDR_194:0x[a-z0-9]*]] <line:81:50, col:56> 'int'
// CHECK-NEXT:         | | | | | | | | | |   |-CallExpr [[ADDR_195:0x[a-z0-9]*]] <col:50, col:56> 'int'
// CHECK-NEXT:         | | | | | | | | | |   | `-ImplicitCastExpr [[ADDR_196:0x[a-z0-9]*]] <col:50> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | |   |   `-DeclRefExpr [[ADDR_197:0x[a-z0-9]*]] <col:50> 'int ({{.*}})' {{.*}}Function [[ADDR_58]] 'base5' 'int ({{.*}})'
// CHECK-NEXT:         | | | | | | | | | |   `-CallExpr [[ADDR_198:0x[a-z0-9]*]] <line:31:29, line:81:56> 'int'
// CHECK-NEXT:         | | | | | | | | | |     `-ImplicitCastExpr [[ADDR_199:0x[a-z0-9]*]] <line:31:29> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | | |       `-DeclRefExpr [[ADDR_63]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_10]] 'picked5' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT:         | | | | | | | | | `-CallExpr [[ADDR_200:0x[a-z0-9]*]] <line:81:60, col:66> 'int'
// CHECK-NEXT:         | | | | | | | | |   `-ImplicitCastExpr [[ADDR_201:0x[a-z0-9]*]] <col:60> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | | |     `-DeclRefExpr [[ADDR_202:0x[a-z0-9]*]] <col:60> 'int ({{.*}})' {{.*}}Function [[ADDR_64]] 'base6' 'int ({{.*}})'
// CHECK-NEXT:         | | | | | | | | `-CallExpr [[ADDR_203:0x[a-z0-9]*]] <col:70, col:76> 'int'
// CHECK-NEXT:         | | | | | | | |   `-ImplicitCastExpr [[ADDR_204:0x[a-z0-9]*]] <col:70> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | |     `-DeclRefExpr [[ADDR_205:0x[a-z0-9]*]] <col:70> 'int ({{.*}})' {{.*}}Function [[ADDR_70]] 'base7' 'int ({{.*}})'
// CHECK-NEXT:         | | | | | | | `-PseudoObjectExpr [[ADDR_206:0x[a-z0-9]*]] <line:82:10, col:16> 'int'
// CHECK-NEXT:         | | | | | | |   |-CallExpr [[ADDR_207:0x[a-z0-9]*]] <col:10, col:16> 'int'
// CHECK-NEXT:         | | | | | | |   | `-ImplicitCastExpr [[ADDR_208:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | |   |   `-DeclRefExpr [[ADDR_209:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_76]] 'base8' 'int ({{.*}})'
// CHECK-NEXT:         | | | | | | |   `-CallExpr [[ADDR_210:0x[a-z0-9]*]] <line:40:29, line:82:16> 'int'
// CHECK-NEXT:         | | | | | | |     `-ImplicitCastExpr [[ADDR_211:0x[a-z0-9]*]] <line:40:29> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | |       `-DeclRefExpr [[ADDR_81]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_30]] 'not_picked3' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT:         | | | | | | `-CallExpr [[ADDR_212:0x[a-z0-9]*]] <line:82:20, col:26> 'int'
// CHECK-NEXT:         | | | | | |   `-ImplicitCastExpr [[ADDR_213:0x[a-z0-9]*]] <col:20> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | |     `-DeclRefExpr [[ADDR_214:0x[a-z0-9]*]] <col:20> 'int ({{.*}})' {{.*}}Function [[ADDR_82]] 'base9' 'int ({{.*}})'
// CHECK-NEXT:         | | | | | `-CallExpr [[ADDR_215:0x[a-z0-9]*]] <col:30, col:37> 'int'
// CHECK-NEXT:         | | | | |   `-ImplicitCastExpr [[ADDR_216:0x[a-z0-9]*]] <col:30> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | |     `-DeclRefExpr [[ADDR_217:0x[a-z0-9]*]] <col:30> 'int ({{.*}})' {{.*}}Function [[ADDR_88]] 'base10' 'int ({{.*}})'
// CHECK-NEXT:         | | | | `-CallExpr [[ADDR_218:0x[a-z0-9]*]] <col:41, col:48> 'int'
// CHECK-NEXT:         | | | |   `-ImplicitCastExpr [[ADDR_219:0x[a-z0-9]*]] <col:41> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | |     `-DeclRefExpr [[ADDR_220:0x[a-z0-9]*]] <col:41> 'int ({{.*}})' {{.*}}Function [[ADDR_94]] 'base11' 'int ({{.*}})'
// CHECK-NEXT:         | | | `-PseudoObjectExpr [[ADDR_221:0x[a-z0-9]*]] <col:52, col:59> 'int'
// CHECK-NEXT:         | | |   |-CallExpr [[ADDR_222:0x[a-z0-9]*]] <col:52, col:59> 'int'
// CHECK-NEXT:         | | |   | `-ImplicitCastExpr [[ADDR_223:0x[a-z0-9]*]] <col:52> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | |   |   `-DeclRefExpr [[ADDR_224:0x[a-z0-9]*]] <col:52> 'int ({{.*}})' {{.*}}Function [[ADDR_100]] 'base12' 'int ({{.*}})'
// CHECK-NEXT:         | | |   `-CallExpr [[ADDR_225:0x[a-z0-9]*]] <line:52:29, line:82:59> 'int'
// CHECK-NEXT:         | | |     `-ImplicitCastExpr [[ADDR_226:0x[a-z0-9]*]] <line:52:29> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | |       `-DeclRefExpr [[ADDR_105]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_14]] 'picked6' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT:         | | `-PseudoObjectExpr [[ADDR_227:0x[a-z0-9]*]] <line:82:63, col:70> 'int'
// CHECK-NEXT:         | |   |-CallExpr [[ADDR_228:0x[a-z0-9]*]] <col:63, col:70> 'int'
// CHECK-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_229:0x[a-z0-9]*]] <col:63> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | |   |   `-DeclRefExpr [[ADDR_230:0x[a-z0-9]*]] <col:63> 'int ({{.*}})' {{.*}}Function [[ADDR_106]] 'base13' 'int ({{.*}})'
// CHECK-NEXT:         | |   `-CallExpr [[ADDR_231:0x[a-z0-9]*]] <line:55:29, line:82:70> 'int'
// CHECK-NEXT:         | |     `-ImplicitCastExpr [[ADDR_232:0x[a-z0-9]*]] <line:55:29> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | |       `-DeclRefExpr [[ADDR_111]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_18]] 'picked7' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT:         | `-PseudoObjectExpr [[ADDR_233:0x[a-z0-9]*]] <line:83:10, col:22> 'int'
// CHECK-NEXT:         |   |-CallExpr [[ADDR_234:0x[a-z0-9]*]] <col:10, col:22> 'int'
// CHECK-NEXT:         |   | `-ImplicitCastExpr [[ADDR_235:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |   |   `-DeclRefExpr [[ADDR_236:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_112]] 'overloaded1' 'int ({{.*}})'
// CHECK-NEXT:         |   `-CallExpr [[ADDR_237:0x[a-z0-9]*]] <line:59:1, line:83:22> 'int'
// CHECK-NEXT:         |     `-ImplicitCastExpr [[ADDR_238:0x[a-z0-9]*]] <line:59:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |       `-DeclRefExpr [[ADDR_114]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_115]] 'overloaded1[implementation={extension(match_any)}, device={kind(cpu, gpu)}]' 'int ({{.*}})'
// CHECK-NEXT:         `-PseudoObjectExpr [[ADDR_239:0x[a-z0-9]*]] <line:83:26, col:38> 'int'
// CHECK-NEXT:           |-CallExpr [[ADDR_240:0x[a-z0-9]*]] <col:26, col:38> 'int'
// CHECK-NEXT:           | `-ImplicitCastExpr [[ADDR_241:0x[a-z0-9]*]] <col:26> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:           |   `-DeclRefExpr [[ADDR_242:0x[a-z0-9]*]] <col:26> 'int ({{.*}})' {{.*}}Function [[ADDR_119]] 'overloaded2' 'int ({{.*}})'
// CHECK-NEXT:           `-CallExpr [[ADDR_243:0x[a-z0-9]*]] <line:64:1, line:83:38> 'int'
// CHECK-NEXT:             `-ImplicitCastExpr [[ADDR_244:0x[a-z0-9]*]] <line:64:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:               `-DeclRefExpr [[ADDR_124]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_125]] 'overloaded2[implementation={extension(match_none)}, device={kind(fpga, gpu)}]' 'int ({{.*}})'
