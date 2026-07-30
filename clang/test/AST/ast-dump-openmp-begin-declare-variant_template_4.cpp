// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

template <typename A, typename B>
int template_number_mismatch_1() {
  return 0;
}

template <typename A, typename B>
int template_number_mismatch_2() {
  return 1;
}

#pragma omp begin declare variant match(implementation = {extension(allow_templates)})
template <typename Q>
int template_number_mismatch_1() {
  return 2;
}
template <typename Q>
int template_number_mismatch_2() {
  return 0;
}
#pragma omp end declare variant

int test() {
  // Should return 0.
  return template_number_mismatch_1<int, float>() + template_number_mismatch_2<double>();
}

// CHECK:      |-FunctionTemplateDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:8:1> line:6:5 template_number_mismatch_1
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_1:0x[a-z0-9]*]] <line:5:11, col:20> col:20 typename depth 0 index 0 A
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_2:0x[a-z0-9]*]] <col:23, col:32> col:32 typename depth 0 index 1 B
// CHECK-NEXT: | |-FunctionDecl [[ADDR_3:0x[a-z0-9]*]] <line:6:1, line:8:1> line:6:5 template_number_mismatch_1 'int ({{.*}})'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_4:0x[a-z0-9]*]] <col:34, line:8:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_5:0x[a-z0-9]*]] <line:7:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_6:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:6:1, line:8:1> line:6:5 used template_number_mismatch_1 'int ({{.*}})'
// CHECK-NEXT: |   |-TemplateArgument type 'int'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_8:0x[a-z0-9]*]] 'int'
// CHECK-NEXT: |   |-TemplateArgument type 'float'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_9:0x[a-z0-9]*]] 'float'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_10:0x[a-z0-9]*]] <col:34, line:8:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_11:0x[a-z0-9]*]] <line:7:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_6]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_12:0x[a-z0-9]*]] <line:10:1, line:13:1> line:11:5 template_number_mismatch_2
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_13:0x[a-z0-9]*]] <line:10:11, col:20> col:20 typename depth 0 index 0 A
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_14:0x[a-z0-9]*]] <col:23, col:32> col:32 typename depth 0 index 1 B
// CHECK-NEXT: | `-FunctionDecl [[ADDR_15:0x[a-z0-9]*]] <line:11:1, line:13:1> line:11:5 template_number_mismatch_2 'int ({{.*}})'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_16:0x[a-z0-9]*]] <col:34, line:13:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_17:0x[a-z0-9]*]] <line:12:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_18:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_19:0x[a-z0-9]*]] <line:16:1, line:17:32> col:5 implicit template_number_mismatch_1
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_20:0x[a-z0-9]*]] <line:16:11, col:20> col:20 typename depth 0 index 0 Q
// CHECK-NEXT: | `-FunctionDecl [[ADDR_21:0x[a-z0-9]*]] <line:17:1, col:32> col:5 template_number_mismatch_1 'int ({{.*}})'
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_22:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_23:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_24:0x[a-z0-9]*]] 'template_number_mismatch_1[implementation={extension(allow_templates)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_25:0x[a-z0-9]*]] <line:16:1, line:19:1> line:17:1 template_number_mismatch_1[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_20]] <line:16:11, col:20> col:20 typename depth 0 index 0 Q
// CHECK-NEXT: | `-FunctionDecl [[ADDR_24]] <line:17:1, line:19:1> line:17:1 template_number_mismatch_1[implementation={extension(allow_templates)}] 'int ({{.*}})'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_26:0x[a-z0-9]*]] <col:34, line:19:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_27:0x[a-z0-9]*]] <line:18:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_29:0x[a-z0-9]*]] <line:20:1, line:21:32> col:5 implicit template_number_mismatch_2
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_30:0x[a-z0-9]*]] <line:20:11, col:20> col:20 typename depth 0 index 0 Q
// CHECK-NEXT: | |-FunctionDecl [[ADDR_31:0x[a-z0-9]*]] <line:21:1, col:32> col:5 template_number_mismatch_2 'int ({{.*}})'
// CHECK-NEXT: | | `-OMPDeclareVariantAttr [[ADDR_32:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: | |   `-DeclRefExpr [[ADDR_33:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_34:0x[a-z0-9]*]] 'template_number_mismatch_2[implementation={extension(allow_templates)}]' 'int ({{.*}})'
// CHECK-NEXT: | `-FunctionDecl [[ADDR_35:0x[a-z0-9]*]] <col:1, col:32> col:5 used template_number_mismatch_2 'int ({{.*}})'
// CHECK-NEXT: |   |-TemplateArgument type 'double'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_36:0x[a-z0-9]*]] 'double'
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_37:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_38:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_39:0x[a-z0-9]*]] 'template_number_mismatch_2[implementation={extension(allow_templates)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_40:0x[a-z0-9]*]] <line:20:1, line:23:1> line:21:1 template_number_mismatch_2[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_30]] <line:20:11, col:20> col:20 typename depth 0 index 0 Q
// CHECK-NEXT: | |-FunctionDecl [[ADDR_34]] <line:21:1, line:23:1> line:21:1 referenced template_number_mismatch_2[implementation={extension(allow_templates)}] 'int ({{.*}})'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_41:0x[a-z0-9]*]] <col:34, line:23:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_42:0x[a-z0-9]*]] <line:22:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_43:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_39]] <line:21:1, line:23:1> line:21:1 template_number_mismatch_2[implementation={extension(allow_templates)}] 'int ({{.*}})'
// CHECK-NEXT: |   |-TemplateArgument type 'double'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_36]] 'double'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_44:0x[a-z0-9]*]] <col:34, line:23:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_45:0x[a-z0-9]*]] <line:22:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_43]] <col:10> 'int' 0
// CHECK-NEXT: `-FunctionDecl [[ADDR_46:0x[a-z0-9]*]] <line:26:1, line:29:1> line:26:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_47:0x[a-z0-9]*]] <col:12, line:29:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_48:0x[a-z0-9]*]] <line:28:3, col:88>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_49:0x[a-z0-9]*]] <col:10, col:88> 'int' '+'
// CHECK-NEXT:         |-CallExpr [[ADDR_50:0x[a-z0-9]*]] <col:10, col:49> 'int'
// CHECK-NEXT:         | `-ImplicitCastExpr [[ADDR_51:0x[a-z0-9]*]] <col:10, col:47> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |   `-DeclRefExpr [[ADDR_52:0x[a-z0-9]*]] <col:10, col:47> 'int ({{.*}})' {{.*}}Function [[ADDR_7]] 'template_number_mismatch_1' 'int ({{.*}})' (FunctionTemplate [[ADDR_0]] 'template_number_mismatch_1')
// CHECK-NEXT:         `-PseudoObjectExpr [[ADDR_53:0x[a-z0-9]*]] <col:53, col:88> 'int'
// CHECK-NEXT:           |-CallExpr [[ADDR_54:0x[a-z0-9]*]] <col:53, col:88> 'int'
// CHECK-NEXT:           | `-ImplicitCastExpr [[ADDR_55:0x[a-z0-9]*]] <col:53, col:86> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:           |   `-DeclRefExpr [[ADDR_56:0x[a-z0-9]*]] <col:53, col:86> 'int ({{.*}})' {{.*}}Function [[ADDR_35]] 'template_number_mismatch_2' 'int ({{.*}})' (FunctionTemplate [[ADDR_29]] 'template_number_mismatch_2')
// CHECK-NEXT:           `-CallExpr [[ADDR_57:0x[a-z0-9]*]] <line:21:1, line:28:88> 'int'
// CHECK-NEXT:             `-ImplicitCastExpr [[ADDR_58:0x[a-z0-9]*]] <line:21:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:               `-DeclRefExpr [[ADDR_38]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_39]] 'template_number_mismatch_2[implementation={extension(allow_templates)}]' 'int ({{.*}})'
