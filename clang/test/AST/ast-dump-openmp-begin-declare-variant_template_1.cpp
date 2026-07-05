// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

int also_before() {
  return 1;
}

#pragma omp begin declare variant match(implementation={vendor(score(100):llvm)})
int also_after(void) {
  return 2;
}
int also_after(int) {
  return 3;
}
int also_after(double) {
  return 0;
}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(0):llvm)})
int also_before() {
  return 0;
}
#pragma omp end declare variant

int also_after(void) {
  return 4;
}
int also_after(int) {
  return 5;
}
int also_after(double) {
  return 6;
}

template<typename T>
int test1() {
  // Should return 0.
  return also_after(T(0));
}

typedef int(*Ty)();

template<Ty fn>
int test2() {
  // Should return 0.
  return fn();
}

int test() {
  // Should return 0.
  return test1<double>() + test2<also_before>();
}

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:19, line:7:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(0): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:21:1> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit also_after 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:12:1> line:10:1 also_after[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:12:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:11:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: |-FunctionDecl [[ADDR_14:0x[a-z0-9]*]] <line:13:1, col:19> col:5 implicit also_after 'int (int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_15:0x[a-z0-9]*]] <col:16> col:19 'int'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_16:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_17:0x[a-z0-9]*]] <col:1> 'int (int)' Function [[ADDR_18:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int (int)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_18]] <col:1, line:15:1> line:13:1 also_after[implementation={vendor(llvm)}] 'int (int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_15]] <col:16> col:19 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_19:0x[a-z0-9]*]] <col:21, line:15:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_20:0x[a-z0-9]*]] <line:14:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_21:0x[a-z0-9]*]] <col:10> 'int' 3
// CHECK-NEXT: |-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:16:1, col:22> col:5 implicit used also_after 'int (double)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_23:0x[a-z0-9]*]] <col:16> col:22 'double'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_24:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_25:0x[a-z0-9]*]] <col:1> 'int (double)' Function [[ADDR_26:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int (double)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_26]] <col:1, line:18:1> line:16:1 also_after[implementation={vendor(llvm)}] 'int (double)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_23]] <col:16> col:22 'double'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_27:0x[a-z0-9]*]] <col:24, line:18:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_28:0x[a-z0-9]*]] <line:17:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_29:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_6]] <line:21:1, line:23:1> line:21:1 also_before[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_30:0x[a-z0-9]*]] <col:19, line:23:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_31:0x[a-z0-9]*]] <line:22:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_32:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_33:0x[a-z0-9]*]] prev [[ADDR_7]] <line:26:1, line:28:1> line:26:5 also_after 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_34:0x[a-z0-9]*]] <col:22, line:28:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_35:0x[a-z0-9]*]] <line:27:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_36:0x[a-z0-9]*]] <col:10> 'int' 4
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_37:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:10:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_38:0x[a-z0-9]*]] prev [[ADDR_14]] <line:29:1, line:31:1> line:29:5 also_after 'int (int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_39:0x[a-z0-9]*]] <col:16> col:19 'int'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_40:0x[a-z0-9]*]] <col:21, line:31:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_41:0x[a-z0-9]*]] <line:30:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_42:0x[a-z0-9]*]] <col:10> 'int' 5
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_43:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_17]] <line:13:1> 'int (int)' Function [[ADDR_18]] 'also_after[implementation={vendor(llvm)}]' 'int (int)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_44:0x[a-z0-9]*]] prev [[ADDR_22]] <line:32:1, line:34:1> line:32:5 used also_after 'int (double)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_45:0x[a-z0-9]*]] <col:16> col:22 'double'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_46:0x[a-z0-9]*]] <col:24, line:34:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_47:0x[a-z0-9]*]] <line:33:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_48:0x[a-z0-9]*]] <col:10> 'int' 6
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_49:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_25]] <line:16:1> 'int (double)' Function [[ADDR_26]] 'also_after[implementation={vendor(llvm)}]' 'int (double)'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_50:0x[a-z0-9]*]] <line:36:1, line:40:1> line:37:5 test1
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_51:0x[a-z0-9]*]] <line:36:10, col:19> col:19 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_52:0x[a-z0-9]*]] <line:37:1, line:40:1> line:37:5 test1 'int ({{.*}})'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_53:0x[a-z0-9]*]] <col:13, line:40:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_54:0x[a-z0-9]*]] <line:39:3, col:25>
// CHECK-NEXT: | |     `-CallExpr [[ADDR_55:0x[a-z0-9]*]] <col:10, col:25> '<dependent type>'
// CHECK-NEXT: | |       |-UnresolvedLookupExpr [[ADDR_56:0x[a-z0-9]*]] <col:10> '<overloaded function type>' {{.*}}(ADL) = 'also_after' [[ADDR_44]] [[ADDR_38]] [[ADDR_33]]
// CHECK-NEXT: | |       `-CXXUnresolvedConstructExpr [[ADDR_57:0x[a-z0-9]*]] <col:21, col:24> 'T' 'T'
// CHECK-NEXT: | |         `-IntegerLiteral [[ADDR_58:0x[a-z0-9]*]] <col:23> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_59:0x[a-z0-9]*]] <line:37:1, line:40:1> line:37:5 used test1 'int ({{.*}})'
// CHECK-NEXT: |   |-TemplateArgument type 'double'
// CHECK:      |   `-CompoundStmt [[ADDR_60:0x[a-z0-9]*]] <col:13, line:40:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_61:0x[a-z0-9]*]] <line:39:3, col:25>
// CHECK-NEXT: |       `-PseudoObjectExpr [[ADDR_62:0x[a-z0-9]*]] <col:10, col:25> 'int'
// CHECK-NEXT: |         |-CallExpr [[ADDR_63:0x[a-z0-9]*]] <col:10, col:25> 'int'
// CHECK-NEXT: |         | |-ImplicitCastExpr [[ADDR_64:0x[a-z0-9]*]] <col:10> 'int (*)(double)' <FunctionToPointerDecay>
// CHECK-NEXT: |         | | `-DeclRefExpr [[ADDR_65:0x[a-z0-9]*]] <col:10> 'int (double)' {{.*}}Function [[ADDR_44]] 'also_after' 'int (double)'
// CHECK-NEXT: |         | `-CXXFunctionalCastExpr [[ADDR_66:0x[a-z0-9]*]] <col:21, col:24> 'double' functional cast to double <NoOp>
// CHECK-NEXT: |         |   `-ImplicitCastExpr [[ADDR_67:0x[a-z0-9]*]] <col:23> 'double' <IntegralToFloating> part_of_explicit_cast
// CHECK-NEXT: |         |     `-IntegerLiteral [[ADDR_58]] <col:23> 'int' 0
// CHECK-NEXT: |         `-CallExpr [[ADDR_68:0x[a-z0-9]*]] <line:16:1, line:39:25> 'int'
// CHECK-NEXT: |           |-ImplicitCastExpr [[ADDR_69:0x[a-z0-9]*]] <line:16:1> 'int (*)(double)' <FunctionToPointerDecay>
// CHECK-NEXT: |           | `-DeclRefExpr [[ADDR_25]] <col:1> 'int (double)' Function [[ADDR_26]] 'also_after[implementation={vendor(llvm)}]' 'int (double)'
// CHECK-NEXT: |           `-CXXFunctionalCastExpr [[ADDR_66]] <line:39:21, col:24> 'double' functional cast to double <NoOp>
// CHECK-NEXT: |             `-ImplicitCastExpr [[ADDR_67]] <col:23> 'double' <IntegralToFloating> part_of_explicit_cast
// CHECK-NEXT: |               `-IntegerLiteral [[ADDR_58]] <col:23> 'int' 0
// CHECK-NEXT: |-TypedefDecl [[ADDR_70:0x[a-z0-9]*]] <line:42:1, col:18> col:14 referenced Ty 'int (*)({{.*}})'
// CHECK-NEXT: | `-PointerType [[ADDR_71:0x[a-z0-9]*]] 'int (*)({{.*}})'
// CHECK-NEXT: |   `-ParenType [[ADDR_72:0x[a-z0-9]*]] 'int ({{.*}})' sugar
// CHECK-NEXT: |     `-FunctionProtoType [[ADDR_73:0x[a-z0-9]*]] 'int ({{.*}})' cdecl
// CHECK-NEXT: |       `-BuiltinType [[ADDR_74:0x[a-z0-9]*]] 'int'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_75:0x[a-z0-9]*]] <line:44:1, line:48:1> line:45:5 test2
// CHECK-NEXT: | |-NonTypeTemplateParmDecl [[ADDR_76:0x[a-z0-9]*]] <line:44:10, col:13> col:13 referenced 'Ty':'int (*)({{.*}})' depth 0 index 0 fn
// CHECK-NEXT: | |-FunctionDecl [[ADDR_77:0x[a-z0-9]*]] <line:45:1, line:48:1> line:45:5 test2 'int ({{.*}})'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_78:0x[a-z0-9]*]] <col:13, line:48:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_79:0x[a-z0-9]*]] <line:47:3, col:13>
// CHECK-NEXT: | |     `-CallExpr [[ADDR_80:0x[a-z0-9]*]] <col:10, col:13> 'int'
// CHECK-NEXT: | |       `-DeclRefExpr [[ADDR_81:0x[a-z0-9]*]] <col:10> 'Ty':'int (*)({{.*}})' NonTypeTemplateParm [[ADDR_76]] 'fn' 'Ty':'int (*)({{.*}})'
// CHECK-NEXT: | `-FunctionDecl [[ADDR_82:0x[a-z0-9]*]] <line:45:1, line:48:1> line:45:5 used test2 'int ({{.*}})'
// CHECK-NEXT: |   |-TemplateArgument decl
// CHECK-NEXT: |   | `-Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_83:0x[a-z0-9]*]] <col:13, line:48:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_84:0x[a-z0-9]*]] <line:47:3, col:13>
// CHECK-NEXT: |       `-PseudoObjectExpr [[ADDR_85:0x[a-z0-9]*]] <col:10, col:13> 'int'
// CHECK-NEXT: |         |-CallExpr [[ADDR_86:0x[a-z0-9]*]] <col:10, col:13> 'int'
// CHECK-NEXT: |         | `-SubstNonTypeTemplateParmExpr [[ADDR_87:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})'
// CHECK-NEXT: |         |   |-NonTypeTemplateParmDecl {{.*}} referenced 'Ty':'int (*)()' depth 0 index 0 fn
// CHECK-NEXT: |         |   `-UnaryOperator [[ADDR_88:0x[a-z0-9]*]] <line:47:10> 'int (*)({{.*}})' prefix '&' cannot overflow
// CHECK-NEXT: |         |     `-DeclRefExpr [[ADDR_89:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CHECK-NEXT: |         `-CallExpr [[ADDR_90:0x[a-z0-9]*]] <line:21:1, line:47:13> 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr [[ADDR_91:0x[a-z0-9]*]] <line:21:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |             `-DeclRefExpr [[ADDR_5]] <col:1> 'int ({{.*}})' Function [[ADDR_6]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: `-FunctionDecl [[ADDR_92:0x[a-z0-9]*]] <line:50:1, line:53:1> line:50:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_93:0x[a-z0-9]*]] <col:12, line:53:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_94:0x[a-z0-9]*]] <line:52:3, col:47>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_95:0x[a-z0-9]*]] <col:10, col:47> 'int' '+'
// CHECK-NEXT:         |-CallExpr [[ADDR_96:0x[a-z0-9]*]] <col:10, col:24> 'int'
// CHECK-NEXT:         | `-ImplicitCastExpr [[ADDR_97:0x[a-z0-9]*]] <col:10, col:22> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |   `-DeclRefExpr [[ADDR_98:0x[a-z0-9]*]] <col:10, col:22> 'int ({{.*}})' {{.*}}Function [[ADDR_59]] 'test1' 'int ({{.*}})' (FunctionTemplate [[ADDR_50]] 'test1')
// CHECK-NEXT:         `-CallExpr [[ADDR_99:0x[a-z0-9]*]] <col:28, col:47> 'int'
// CHECK-NEXT:           `-ImplicitCastExpr [[ADDR_100:0x[a-z0-9]*]] <col:28, col:45> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:             `-DeclRefExpr [[ADDR_101:0x[a-z0-9]*]] <col:28, col:45> 'int ({{.*}})' {{.*}}Function [[ADDR_82]] 'test2' 'int ({{.*}})' (FunctionTemplate [[ADDR_75]] 'test2')
