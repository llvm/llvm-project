// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++ | FileCheck %s
// expected-no-diagnostics

template <typename T>
int also_before(T) {
  return 1;
}
template <int V>
int also_before_mismatch(void) {
  return 0;
}
int also_before_non_template(void) {
  return 0;
}

#pragma omp begin declare variant match(implementation = {extension(allow_templates)})
template <typename T>
int also_before(T) {
  return 0;
}
template <typename T>
int also_after(T) {
  return 0;
}
template <typename T, typename Q>
int also_after_mismatch(T, Q) {
  return 2;
}
template <typename T>
int also_before_mismatch(T) {
  return 3;
}
template <typename T>
int also_before_non_template(T) {
  return 4;
}
template <int V>
int only_def(void) {
  return 0;
}
#pragma omp end declare variant

template <typename T>
int also_after(T) {
  return 6;
}
template <typename T>
int also_after_mismatch(T) {
  return 0;
}

int test() {
  // Should return 0.
  return also_before(0.) + also_before_mismatch<0>() + also_before_non_template() + also_after<char>(0) + also_after_mismatch(0) + only_def<0>();
}

// CHECK:      |-FunctionTemplateDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 also_before
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_1:0x[a-z0-9]*]] <line:4:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_2:0x[a-z0-9]*]] <line:5:1, line:7:1> line:5:5 also_before 'int (T)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_3:0x[a-z0-9]*]] <col:17> col:18 'T'
// CHECK-NEXT: | | |-CompoundStmt [[ADDR_4:0x[a-z0-9]*]] <col:20, line:7:1>
// CHECK-NEXT: | | | `-ReturnStmt [[ADDR_5:0x[a-z0-9]*]] <line:6:3, col:10>
// CHECK-NEXT: | | |   `-IntegerLiteral [[ADDR_6:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: | | `-OMPDeclareVariantAttr [[ADDR_7:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: | |   `-DeclRefExpr [[ADDR_8:0x[a-z0-9]*]] <line:18:1> 'int (T)' {{.*}}Function [[ADDR_9:0x[a-z0-9]*]] 'also_before[implementation={extension(allow_templates)}]' 'int (T)'
// CHECK-NEXT: | `-FunctionDecl [[ADDR_10:0x[a-z0-9]*]] <line:5:1, line:7:1> line:5:5 used also_before 'int (double)'
// CHECK-NEXT: |   |-TemplateArgument type 'double'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_11:0x[a-z0-9]*]] 'double'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_12:0x[a-z0-9]*]] <col:17> col:18 'double'
// CHECK-NEXT: |   |-CompoundStmt [[ADDR_13:0x[a-z0-9]*]] <col:20, line:7:1>
// CHECK-NEXT: |   | `-ReturnStmt [[ADDR_14:0x[a-z0-9]*]] <line:6:3, col:10>
// CHECK-NEXT: |   |   `-IntegerLiteral [[ADDR_6]] <col:10> 'int' 1
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_15:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_16:0x[a-z0-9]*]] <line:18:1> 'int (double)' {{.*}}Function [[ADDR_17:0x[a-z0-9]*]] 'also_before[implementation={extension(allow_templates)}]' 'int (double)'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_18:0x[a-z0-9]*]] <line:8:1, line:11:1> line:9:5 also_before_mismatch
// CHECK-NEXT: | |-NonTypeTemplateParmDecl [[ADDR_19:0x[a-z0-9]*]] <line:8:11, col:15> col:15 'int' depth 0 index 0 V
// CHECK-NEXT: | |-FunctionDecl [[ADDR_20:0x[a-z0-9]*]] <line:9:1, line:11:1> line:9:5 also_before_mismatch 'int ({{.*}})'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_21:0x[a-z0-9]*]] <col:32, line:11:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_22:0x[a-z0-9]*]] <line:10:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_23:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_24:0x[a-z0-9]*]] <line:9:1, line:11:1> line:9:5 used also_before_mismatch 'int ({{.*}})'
// CHECK-NEXT: |   |-TemplateArgument integral '0'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_25:0x[a-z0-9]*]] <col:32, line:11:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_26:0x[a-z0-9]*]] <line:10:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_23]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_27:0x[a-z0-9]*]] <line:12:1, line:14:1> line:12:5 used also_before_non_template 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_28:0x[a-z0-9]*]] <col:36, line:14:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_29:0x[a-z0-9]*]] <line:13:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_30:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_31:0x[a-z0-9]*]] <line:17:1, line:20:1> line:18:1 also_before[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_32:0x[a-z0-9]*]] <line:17:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_9]] <line:18:1, line:20:1> line:18:1 referenced also_before[implementation={extension(allow_templates)}] 'int (T)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_33:0x[a-z0-9]*]] <col:17> col:18 'T'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_34:0x[a-z0-9]*]] <col:20, line:20:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_35:0x[a-z0-9]*]] <line:19:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_36:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_17]] <line:18:1, line:20:1> line:18:1 also_before[implementation={extension(allow_templates)}] 'int (double)'
// CHECK-NEXT: |   |-TemplateArgument type 'double'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_11]] 'double'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_37:0x[a-z0-9]*]] <col:17> col:18 'double'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_38:0x[a-z0-9]*]] <col:20, line:20:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_39:0x[a-z0-9]*]] <line:19:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_36]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_40:0x[a-z0-9]*]] <line:21:1, line:22:17> col:5 implicit also_after
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_41:0x[a-z0-9]*]] <line:21:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_42:0x[a-z0-9]*]] <line:22:1, col:17> col:5 also_after 'int (T)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_43:0x[a-z0-9]*]] <col:16> col:17 'T'
// CHECK-NEXT: | | `-OMPDeclareVariantAttr [[ADDR_44:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: | |   `-DeclRefExpr [[ADDR_45:0x[a-z0-9]*]] <col:1> 'int (T)' {{.*}}Function [[ADDR_46:0x[a-z0-9]*]] 'also_after[implementation={extension(allow_templates)}]' 'int (T)'
// CHECK-NEXT: | `-FunctionDecl [[ADDR_47:0x[a-z0-9]*]] <line:44:1, line:46:1> line:44:5 used also_after 'int (char)'
// CHECK-NEXT: |   |-TemplateArgument type 'char'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_48:0x[a-z0-9]*]] 'char'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_49:0x[a-z0-9]*]] <col:16> col:17 'char'
// CHECK-NEXT: |   |-CompoundStmt [[ADDR_50:0x[a-z0-9]*]] <col:19, line:46:1>
// CHECK-NEXT: |   | `-ReturnStmt [[ADDR_51:0x[a-z0-9]*]] <line:45:3, col:10>
// CHECK-NEXT: |   |   `-IntegerLiteral [[ADDR_52:0x[a-z0-9]*]] <col:10> 'int' 6
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_53:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_54:0x[a-z0-9]*]] <line:22:1> 'int (char)' {{.*}}Function [[ADDR_55:0x[a-z0-9]*]] 'also_after[implementation={extension(allow_templates)}]' 'int (char)'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_56:0x[a-z0-9]*]] <line:21:1, line:24:1> line:22:1 also_after[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_41]] <line:21:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_46]] <line:22:1, line:24:1> line:22:1 referenced also_after[implementation={extension(allow_templates)}] 'int (T)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_43]] <col:16> col:17 'T'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_57:0x[a-z0-9]*]] <col:19, line:24:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_58:0x[a-z0-9]*]] <line:23:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_59:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_55]] <line:22:1, line:24:1> line:22:1 also_after[implementation={extension(allow_templates)}] 'int (char)'
// CHECK-NEXT: |   |-TemplateArgument type 'char'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_48]] 'char'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_60:0x[a-z0-9]*]] <col:16> col:17 'char'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_61:0x[a-z0-9]*]] <col:19, line:24:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_62:0x[a-z0-9]*]] <line:23:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_59]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_63:0x[a-z0-9]*]] <line:25:1, line:26:29> col:5 implicit also_after_mismatch
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_64:0x[a-z0-9]*]] <line:25:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_65:0x[a-z0-9]*]] <col:23, col:32> col:32 referenced typename depth 0 index 1 Q
// CHECK-NEXT: | `-FunctionDecl [[ADDR_66:0x[a-z0-9]*]] <line:26:1, col:29> col:5 also_after_mismatch 'int (T, Q)'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_67:0x[a-z0-9]*]] <col:25> col:26 'T'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_68:0x[a-z0-9]*]] <col:28> col:29 'Q'
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_69:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_70:0x[a-z0-9]*]] <col:1> 'int (T, Q)' {{.*}}Function [[ADDR_71:0x[a-z0-9]*]] 'also_after_mismatch[implementation={extension(allow_templates)}]' 'int (T, Q)'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_72:0x[a-z0-9]*]] <line:25:1, line:28:1> line:26:1 also_after_mismatch[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_64]] <line:25:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_65]] <col:23, col:32> col:32 referenced typename depth 0 index 1 Q
// CHECK-NEXT: | `-FunctionDecl [[ADDR_71]] <line:26:1, line:28:1> line:26:1 also_after_mismatch[implementation={extension(allow_templates)}] 'int (T, Q)'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_67]] <col:25> col:26 'T'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_68]] <col:28> col:29 'Q'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_73:0x[a-z0-9]*]] <col:31, line:28:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_74:0x[a-z0-9]*]] <line:27:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_75:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_76:0x[a-z0-9]*]] <line:29:1, line:30:27> col:5 implicit also_before_mismatch
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_77:0x[a-z0-9]*]] <line:29:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | `-FunctionDecl [[ADDR_78:0x[a-z0-9]*]] <line:30:1, col:27> col:5 also_before_mismatch 'int (T)'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_79:0x[a-z0-9]*]] <col:26> col:27 'T'
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_80:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_81:0x[a-z0-9]*]] <col:1> 'int (T)' {{.*}}Function [[ADDR_82:0x[a-z0-9]*]] 'also_before_mismatch[implementation={extension(allow_templates)}]' 'int (T)'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_83:0x[a-z0-9]*]] <line:29:1, line:32:1> line:30:1 also_before_mismatch[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_77]] <line:29:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | `-FunctionDecl [[ADDR_82]] <line:30:1, line:32:1> line:30:1 also_before_mismatch[implementation={extension(allow_templates)}] 'int (T)'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_79]] <col:26> col:27 'T'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_84:0x[a-z0-9]*]] <col:29, line:32:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_85:0x[a-z0-9]*]] <line:31:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_86:0x[a-z0-9]*]] <col:10> 'int' 3
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_87:0x[a-z0-9]*]] <line:33:1, line:34:31> col:5 implicit also_before_non_template
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_88:0x[a-z0-9]*]] <line:33:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | `-FunctionDecl [[ADDR_89:0x[a-z0-9]*]] <line:34:1, col:31> col:5 also_before_non_template 'int (T)'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_90:0x[a-z0-9]*]] <col:30> col:31 'T'
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_91:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_92:0x[a-z0-9]*]] <col:1> 'int (T)' {{.*}}Function [[ADDR_93:0x[a-z0-9]*]] 'also_before_non_template[implementation={extension(allow_templates)}]' 'int (T)'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_94:0x[a-z0-9]*]] <line:33:1, line:36:1> line:34:1 also_before_non_template[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_88]] <line:33:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | `-FunctionDecl [[ADDR_93]] <line:34:1, line:36:1> line:34:1 also_before_non_template[implementation={extension(allow_templates)}] 'int (T)'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_90]] <col:30> col:31 'T'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_95:0x[a-z0-9]*]] <col:33, line:36:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_96:0x[a-z0-9]*]] <line:35:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_97:0x[a-z0-9]*]] <col:10> 'int' 4
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_98:0x[a-z0-9]*]] <line:37:1, line:38:18> col:5 implicit only_def
// CHECK-NEXT: | |-NonTypeTemplateParmDecl [[ADDR_99:0x[a-z0-9]*]] <line:37:11, col:15> col:15 'int' depth 0 index 0 V
// CHECK-NEXT: | |-FunctionDecl [[ADDR_100:0x[a-z0-9]*]] <line:38:1, col:18> col:5 only_def 'int ({{.*}})'
// CHECK-NEXT: | | `-OMPDeclareVariantAttr [[ADDR_101:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: | |   `-DeclRefExpr [[ADDR_102:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_103:0x[a-z0-9]*]] 'only_def[implementation={extension(allow_templates)}]' 'int ({{.*}})'
// CHECK-NEXT: | `-FunctionDecl [[ADDR_104:0x[a-z0-9]*]] <col:1, col:18> col:5 used only_def 'int ({{.*}})'
// CHECK-NEXT: |   |-TemplateArgument integral '0'
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_105:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_106:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_107:0x[a-z0-9]*]] 'only_def[implementation={extension(allow_templates)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_108:0x[a-z0-9]*]] <line:37:1, line:40:1> line:38:1 only_def[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-NonTypeTemplateParmDecl [[ADDR_99]] <line:37:11, col:15> col:15 'int' depth 0 index 0 V
// CHECK-NEXT: | |-FunctionDecl [[ADDR_103]] <line:38:1, line:40:1> line:38:1 referenced only_def[implementation={extension(allow_templates)}] 'int ({{.*}})'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_109:0x[a-z0-9]*]] <col:20, line:40:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_110:0x[a-z0-9]*]] <line:39:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_111:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_107]] <line:38:1, line:40:1> line:38:1 only_def[implementation={extension(allow_templates)}] 'int ({{.*}})'
// CHECK-NEXT: |   |-TemplateArgument integral '0'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_112:0x[a-z0-9]*]] <col:20, line:40:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_113:0x[a-z0-9]*]] <line:39:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_111]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_114:0x[a-z0-9]*]] prev [[ADDR_40]] <line:43:1, line:46:1> line:44:5 also_after
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_115:0x[a-z0-9]*]] <line:43:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_116:0x[a-z0-9]*]] prev [[ADDR_42]] <line:44:1, line:46:1> line:44:5 also_after 'int (T)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_117:0x[a-z0-9]*]] <col:16> col:17 'T'
// CHECK-NEXT: | | |-CompoundStmt [[ADDR_118:0x[a-z0-9]*]] <col:19, line:46:1>
// CHECK-NEXT: | | | `-ReturnStmt [[ADDR_119:0x[a-z0-9]*]] <line:45:3, col:10>
// CHECK-NEXT: | | |   `-IntegerLiteral [[ADDR_52]] <col:10> 'int' 6
// CHECK-NEXT: | | `-OMPDeclareVariantAttr [[ADDR_120:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: | |   `-DeclRefExpr [[ADDR_45]] <line:22:1> 'int (T)' {{.*}}Function [[ADDR_46]] 'also_after[implementation={extension(allow_templates)}]' 'int (T)'
// CHECK-NEXT: | `-Function [[ADDR_47]] 'also_after' 'int (char)'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_121:0x[a-z0-9]*]] <line:47:1, line:50:1> line:48:5 also_after_mismatch
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_122:0x[a-z0-9]*]] <line:47:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_123:0x[a-z0-9]*]] <line:48:1, line:50:1> line:48:5 also_after_mismatch 'int (T)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_124:0x[a-z0-9]*]] <col:25> col:26 'T'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_125:0x[a-z0-9]*]] <col:28, line:50:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_126:0x[a-z0-9]*]] <line:49:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_127:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_128:0x[a-z0-9]*]] <line:48:1, line:50:1> line:48:5 used also_after_mismatch 'int (int)'
// CHECK-NEXT: |   |-TemplateArgument type 'int'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_129:0x[a-z0-9]*]] 'int'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_130:0x[a-z0-9]*]] <col:25> col:26 'int'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_131:0x[a-z0-9]*]] <col:28, line:50:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_132:0x[a-z0-9]*]] <line:49:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_127]] <col:10> 'int' 0
// CHECK-NEXT: `-FunctionDecl [[ADDR_133:0x[a-z0-9]*]] <line:52:1, line:55:1> line:52:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_134:0x[a-z0-9]*]] <col:12, line:55:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_135:0x[a-z0-9]*]] <line:54:3, col:144>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_136:0x[a-z0-9]*]] <col:10, col:144> 'int' '+'
// CHECK-NEXT:         |-BinaryOperator [[ADDR_137:0x[a-z0-9]*]] <col:10, col:128> 'int' '+'
// CHECK-NEXT:         | |-BinaryOperator [[ADDR_138:0x[a-z0-9]*]] <col:10, col:103> 'int' '+'
// CHECK-NEXT:         | | |-BinaryOperator [[ADDR_139:0x[a-z0-9]*]] <col:10, col:81> 'int' '+'
// CHECK-NEXT:         | | | |-BinaryOperator [[ADDR_140:0x[a-z0-9]*]] <col:10, col:52> 'int' '+'
// CHECK-NEXT:         | | | | |-PseudoObjectExpr [[ADDR_141:0x[a-z0-9]*]] <col:10, col:24> 'int'
// CHECK-NEXT:         | | | | | |-CallExpr [[ADDR_142:0x[a-z0-9]*]] <col:10, col:24> 'int'
// CHECK-NEXT:         | | | | | | |-ImplicitCastExpr [[ADDR_143:0x[a-z0-9]*]] <col:10> 'int (*)(double)' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | | | | `-DeclRefExpr [[ADDR_144:0x[a-z0-9]*]] <col:10> 'int (double)' {{.*}}Function [[ADDR_10]] 'also_before' 'int (double)' (FunctionTemplate [[ADDR_0]] 'also_before')
// CHECK-NEXT:         | | | | | | `-FloatingLiteral [[ADDR_145:0x[a-z0-9]*]] <col:22> 'double' 0.000000e+00
// CHECK-NEXT:         | | | | | `-CallExpr [[ADDR_146:0x[a-z0-9]*]] <line:18:1, line:54:24> 'int'
// CHECK-NEXT:         | | | | |   |-ImplicitCastExpr [[ADDR_147:0x[a-z0-9]*]] <line:18:1> 'int (*)(double)' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | | |   | `-DeclRefExpr [[ADDR_16]] <col:1> 'int (double)' {{.*}}Function [[ADDR_17]] 'also_before[implementation={extension(allow_templates)}]' 'int (double)'
// CHECK-NEXT:         | | | | |   `-FloatingLiteral [[ADDR_145]] <line:54:22> 'double' 0.000000e+00
// CHECK-NEXT:         | | | | `-CallExpr [[ADDR_148:0x[a-z0-9]*]] <col:28, col:52> 'int'
// CHECK-NEXT:         | | | |   `-ImplicitCastExpr [[ADDR_149:0x[a-z0-9]*]] <col:28, col:50> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | | |     `-DeclRefExpr [[ADDR_150:0x[a-z0-9]*]] <col:28, col:50> 'int ({{.*}})' {{.*}}Function [[ADDR_24]] 'also_before_mismatch' 'int ({{.*}})' (FunctionTemplate [[ADDR_18]] 'also_before_mismatch')
// CHECK-NEXT:         | | | `-CallExpr [[ADDR_151:0x[a-z0-9]*]] <col:56, col:81> 'int'
// CHECK-NEXT:         | | |   `-ImplicitCastExpr [[ADDR_152:0x[a-z0-9]*]] <col:56> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | |     `-DeclRefExpr [[ADDR_153:0x[a-z0-9]*]] <col:56> 'int ({{.*}})' {{.*}}Function [[ADDR_27]] 'also_before_non_template' 'int ({{.*}})'
// CHECK-NEXT:         | | `-PseudoObjectExpr [[ADDR_154:0x[a-z0-9]*]] <col:85, col:103> 'int'
// CHECK-NEXT:         | |   |-CallExpr [[ADDR_155:0x[a-z0-9]*]] <col:85, col:103> 'int'
// CHECK-NEXT:         | |   | |-ImplicitCastExpr [[ADDR_156:0x[a-z0-9]*]] <col:85, col:100> 'int (*)(char)' <FunctionToPointerDecay>
// CHECK-NEXT:         | |   | | `-DeclRefExpr [[ADDR_157:0x[a-z0-9]*]] <col:85, col:100> 'int (char)' {{.*}}Function [[ADDR_47]] 'also_after' 'int (char)' (FunctionTemplate [[ADDR_114]] 'also_after')
// CHECK-NEXT:         | |   | `-ImplicitCastExpr [[ADDR_158:0x[a-z0-9]*]] <col:102> 'char' <IntegralCast>
// CHECK-NEXT:         | |   |   `-IntegerLiteral [[ADDR_159:0x[a-z0-9]*]] <col:102> 'int' 0
// CHECK-NEXT:         | |   `-CallExpr [[ADDR_160:0x[a-z0-9]*]] <line:22:1, line:54:103> 'int'
// CHECK-NEXT:         | |     |-ImplicitCastExpr [[ADDR_161:0x[a-z0-9]*]] <line:22:1> 'int (*)(char)' <FunctionToPointerDecay>
// CHECK-NEXT:         | |     | `-DeclRefExpr [[ADDR_54]] <col:1> 'int (char)' {{.*}}Function [[ADDR_55]] 'also_after[implementation={extension(allow_templates)}]' 'int (char)'
// CHECK-NEXT:         | |     `-ImplicitCastExpr [[ADDR_162:0x[a-z0-9]*]] <line:54:102> 'char' <IntegralCast>
// CHECK-NEXT:         | |       `-IntegerLiteral [[ADDR_159]] <col:102> 'int' 0
// CHECK-NEXT:         | `-CallExpr [[ADDR_163:0x[a-z0-9]*]] <col:107, col:128> 'int'
// CHECK-NEXT:         |   |-ImplicitCastExpr [[ADDR_164:0x[a-z0-9]*]] <col:107> 'int (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT:         |   | `-DeclRefExpr [[ADDR_165:0x[a-z0-9]*]] <col:107> 'int (int)' {{.*}}Function [[ADDR_128]] 'also_after_mismatch' 'int (int)' (FunctionTemplate [[ADDR_121]] 'also_after_mismatch')
// CHECK-NEXT:         |   `-IntegerLiteral [[ADDR_166:0x[a-z0-9]*]] <col:127> 'int' 0
// CHECK-NEXT:         `-PseudoObjectExpr [[ADDR_167:0x[a-z0-9]*]] <col:132, col:144> 'int'
// CHECK-NEXT:           |-CallExpr [[ADDR_168:0x[a-z0-9]*]] <col:132, col:144> 'int'
// CHECK-NEXT:           | `-ImplicitCastExpr [[ADDR_169:0x[a-z0-9]*]] <col:132, col:142> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:           |   `-DeclRefExpr [[ADDR_170:0x[a-z0-9]*]] <col:132, col:142> 'int ({{.*}})' {{.*}}Function [[ADDR_104]] 'only_def' 'int ({{.*}})' (FunctionTemplate [[ADDR_98]] 'only_def')
// CHECK-NEXT:           `-CallExpr [[ADDR_171:0x[a-z0-9]*]] <line:38:1, line:54:144> 'int'
// CHECK-NEXT:             `-ImplicitCastExpr [[ADDR_172:0x[a-z0-9]*]] <line:38:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:               `-DeclRefExpr [[ADDR_106]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_107]] 'only_def[implementation={extension(allow_templates)}]' 'int ({{.*}})'
