// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics
// PR47655

template <typename T> struct S {
  S(int, T *) {}
};

template <typename T>
int also_before(T s) {
  return 0;
}

#pragma omp begin declare variant match(implementation = {extension(allow_templates)})
template <typename T>
int also_before(S<T> s) {
  // Ensure there is no error because this is never instantiated.
  double t;
  S<T> q(1, &t);
  return 1;
}
template <typename T>
int special(S<T> s) {
  T t;
  S<T> q(0, &t);
  return 0;
}
template <typename T>
int also_after(S<T> s) {
  // Ensure there is no error because this is never instantiated.
  double t;
  S<T> q(2.0, &t);
  return 2;
}
#pragma omp end declare variant

template <typename T>
int also_after(T s) {
  return 0;
}

int test() {
  // Should return 0.
  return also_before(0) + also_after(0) + also_before(0.) + also_after(0.) + special(S<int>(0, 0));
}

// CHECK:      |-ClassTemplateDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:30 S
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_1:0x[a-z0-9]*]] <col:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-CXXRecordDecl [[ADDR_2:0x[a-z0-9]*]] <col:23, line:7:1> line:5:30 struct S definition
// CHECK-NEXT: | | |-DefinitionData empty standard_layout trivially_copyable has_user_declared_ctor can_const_default_init
// CHECK-NEXT: | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK-NEXT: | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: | | |-CXXRecordDecl [[ADDR_3:0x[a-z0-9]*]] <col:23, col:30> col:30 implicit referenced struct S
// CHECK-NEXT: | | `-CXXConstructorDecl [[ADDR_4:0x[a-z0-9]*]] <line:6:3, col:16> col:3 S<T> 'void (int, T *)'
// CHECK-NEXT: | |   |-ParmVarDecl [[ADDR_5:0x[a-z0-9]*]] <col:5> col:8 'int'
// CHECK-NEXT: | |   |-ParmVarDecl [[ADDR_6:0x[a-z0-9]*]] <col:10, col:12> col:13 'T *'
// CHECK-NEXT: | |   `-CompoundStmt [[ADDR_7:0x[a-z0-9]*]] <col:15, col:16>
// CHECK-NEXT: | |-ClassTemplateSpecializationDecl [[ADDR_8:0x[a-z0-9]*]] <line:5:1, line:7:1> line:5:30 struct S definition
// CHECK-NEXT: | | |-DefinitionData pass_in_registers empty standard_layout trivially_copyable has_user_declared_ctor can_const_default_init
// CHECK-NEXT: | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK-NEXT: | | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT: | | | |-MoveConstructor exists simple trivial
// CHECK-NEXT: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | | `-Destructor simple irrelevant trivial
// CHECK-NEXT: | | |-TemplateArgument type 'int'
// CHECK-NEXT: | | | `-BuiltinType [[ADDR_9:0x[a-z0-9]*]] 'int'
// CHECK-NEXT: | | |-CXXRecordDecl [[ADDR_10:0x[a-z0-9]*]] <col:23, col:30> col:30 implicit struct S
// CHECK-NEXT: | | |-CXXConstructorDecl [[ADDR_11:0x[a-z0-9]*]] <line:6:3, col:16> col:3 used S 'void (int, int *)'
// CHECK-NEXT: | | | |-ParmVarDecl [[ADDR_12:0x[a-z0-9]*]] <col:5> col:8 'int'
// CHECK-NEXT: | | | |-ParmVarDecl [[ADDR_13:0x[a-z0-9]*]] <col:10, col:12> col:13 'int *'
// CHECK-NEXT: | | | `-CompoundStmt [[ADDR_7]] <col:15, col:16>
// CHECK-NEXT: | | |-CXXConstructorDecl [[ADDR_14:0x[a-z0-9]*]] <line:5:30> col:30 implicit constexpr S 'void (const S<int> &)' inline default trivial noexcept-unevaluated [[ADDR_14]]
// CHECK-NEXT: | | | `-ParmVarDecl [[ADDR_15:0x[a-z0-9]*]] <col:30> col:30 'const S<int> &'
// CHECK-NEXT: | | |-CXXConstructorDecl [[ADDR_16:0x[a-z0-9]*]] <col:30> col:30 implicit used constexpr S 'void (S<int> &&) noexcept' inline default trivial
// CHECK-NEXT: | | | |-ParmVarDecl [[ADDR_17:0x[a-z0-9]*]] <col:30> col:30 'S<int> &&'
// CHECK-NEXT: | | | `-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:30>
// CHECK-NEXT: | | `-CXXDestructorDecl [[ADDR_19:0x[a-z0-9]*]] <col:30> col:30 implicit referenced ~S 'void ({{.*}}) noexcept' inline default trivial
// CHECK-NEXT: | `-ClassTemplateSpecializationDecl [[ADDR_20:0x[a-z0-9]*]] <col:1, line:7:1> line:5:30 struct S
// CHECK-NEXT: |   `-TemplateArgument type 'double'
// CHECK-NEXT: |     `-BuiltinType [[ADDR_21:0x[a-z0-9]*]] 'double'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_22:0x[a-z0-9]*]] <line:9:1, line:12:1> line:10:5 also_before
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_23:0x[a-z0-9]*]] <line:9:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_24:0x[a-z0-9]*]] <line:10:1, line:12:1> line:10:5 also_before 'int (T)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_25:0x[a-z0-9]*]] <col:17, col:19> col:19 s 'T'
// CHECK-NEXT: | | |-CompoundStmt [[ADDR_26:0x[a-z0-9]*]] <col:22, line:12:1>
// CHECK-NEXT: | | | `-ReturnStmt [[ADDR_27:0x[a-z0-9]*]] <line:11:3, col:10>
// CHECK-NEXT: | | |   `-IntegerLiteral [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | | `-OMPDeclareVariantAttr [[ADDR_29:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: | |   `-DeclRefExpr [[ADDR_30:0x[a-z0-9]*]] <line:16:1> 'int (S<T>)' {{.*}}Function [[ADDR_31:0x[a-z0-9]*]] 'also_before[implementation={extension(allow_templates)}]' 'int (S<T>)'
// CHECK-NEXT: | |-FunctionDecl [[ADDR_32:0x[a-z0-9]*]] <line:10:1, line:12:1> line:10:5 used also_before 'int (int)'
// CHECK-NEXT: | | |-TemplateArgument type 'int'
// CHECK-NEXT: | | | `-BuiltinType [[ADDR_9]] 'int'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_33:0x[a-z0-9]*]] <col:17, col:19> col:19 s 'int':'int'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_34:0x[a-z0-9]*]] <col:22, line:12:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_35:0x[a-z0-9]*]] <line:11:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_28]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_36:0x[a-z0-9]*]] <line:10:1, line:12:1> line:10:5 used also_before 'int (double)'
// CHECK-NEXT: |   |-TemplateArgument type 'double'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_21]] 'double'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_37:0x[a-z0-9]*]] <col:17, col:19> col:19 s 'double':'double'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_38:0x[a-z0-9]*]] <col:22, line:12:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_39:0x[a-z0-9]*]] <line:11:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_28]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_40:0x[a-z0-9]*]] <line:15:1, line:21:1> line:16:1 also_before[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_41:0x[a-z0-9]*]] <line:15:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_31]] <line:16:1, line:21:1> line:16:1 referenced also_before[implementation={extension(allow_templates)}] 'int (S<T>)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_42:0x[a-z0-9]*]] <col:17, col:22> col:22 s 'S<T>'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_43:0x[a-z0-9]*]] <col:25, line:21:1>
// CHECK-NEXT: | |   |-DeclStmt [[ADDR_44:0x[a-z0-9]*]] <line:18:3, col:11>
// CHECK-NEXT: | |   | `-VarDecl [[ADDR_45:0x[a-z0-9]*]] <col:3, col:10> col:10 referenced t 'double'
// CHECK-NEXT: | |   |-DeclStmt [[ADDR_46:0x[a-z0-9]*]] <line:19:3, col:16>
// CHECK-NEXT: | |   | `-VarDecl [[ADDR_47:0x[a-z0-9]*]] <col:3, col:15> col:8 q 'S<T>' callinit
// CHECK-NEXT: | |   |   `-ParenListExpr [[ADDR_48:0x[a-z0-9]*]] <col:9, col:15> 'NULL TYPE'
// CHECK-NEXT: | |   |     |-IntegerLiteral [[ADDR_49:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: | |   |     `-UnaryOperator [[ADDR_50:0x[a-z0-9]*]] <col:13, col:14> 'double *' prefix '&' cannot overflow
// CHECK-NEXT: | |   |       `-DeclRefExpr [[ADDR_51:0x[a-z0-9]*]] <col:14> 'double' {{.*}}Var [[ADDR_45]] 't' 'double'
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_52:0x[a-z0-9]*]] <line:20:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_53:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: | |-FunctionDecl [[ADDR_54:0x[a-z0-9]*]] <line:16:1, line:21:1> line:16:1 also_before[implementation={extension(allow_templates)}] 'int (S<int>)'
// CHECK-NEXT: | | |-TemplateArgument type 'int'
// CHECK-NEXT: | | | `-BuiltinType [[ADDR_9]] 'int'
// CHECK-NEXT: | | `-ParmVarDecl [[ADDR_55:0x[a-z0-9]*]] <col:17, col:22> col:22 s 'S<int>':'S<int>'
// CHECK-NEXT: | `-FunctionDecl [[ADDR_56:0x[a-z0-9]*]] <col:1, line:21:1> line:16:1 also_before[implementation={extension(allow_templates)}] 'int (S<double>)'
// CHECK-NEXT: |   |-TemplateArgument type 'double'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_21]] 'double'
// CHECK-NEXT: |   `-ParmVarDecl [[ADDR_57:0x[a-z0-9]*]] <col:17, col:22> col:22 s 'S<double>':'S<double>'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_58:0x[a-z0-9]*]] <line:22:1, line:23:19> col:5 implicit special
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_59:0x[a-z0-9]*]] <line:22:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_60:0x[a-z0-9]*]] <line:23:1, col:19> col:5 special 'int (S<T>)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_61:0x[a-z0-9]*]] <col:13, col:18> col:18 s 'S<T>'
// CHECK-NEXT: | | `-OMPDeclareVariantAttr [[ADDR_62:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: | |   `-DeclRefExpr [[ADDR_63:0x[a-z0-9]*]] <col:1> 'int (S<T>)' {{.*}}Function [[ADDR_64:0x[a-z0-9]*]] 'special[implementation={extension(allow_templates)}]' 'int (S<T>)'
// CHECK-NEXT: | `-FunctionDecl [[ADDR_65:0x[a-z0-9]*]] <col:1, col:19> col:5 used special 'int (S<int>)'
// CHECK-NEXT: |   |-TemplateArgument type 'int'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_9]] 'int'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_66:0x[a-z0-9]*]] <col:13, col:18> col:18 s 'S<int>':'S<int>'
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_67:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_68:0x[a-z0-9]*]] <col:1> 'int (S<int>)' {{.*}}Function [[ADDR_69:0x[a-z0-9]*]] 'special[implementation={extension(allow_templates)}]' 'int (S<int>)'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_70:0x[a-z0-9]*]] <line:22:1, line:27:1> line:23:1 special[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_59]] <line:22:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_64]] <line:23:1, line:27:1> line:23:1 referenced special[implementation={extension(allow_templates)}] 'int (S<T>)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_61]] <col:13, col:18> col:18 s 'S<T>'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_71:0x[a-z0-9]*]] <col:21, line:27:1>
// CHECK-NEXT: | |   |-DeclStmt [[ADDR_72:0x[a-z0-9]*]] <line:24:3, col:6>
// CHECK-NEXT: | |   | `-VarDecl [[ADDR_73:0x[a-z0-9]*]] <col:3, col:5> col:5 referenced t 'T'
// CHECK-NEXT: | |   |-DeclStmt [[ADDR_74:0x[a-z0-9]*]] <line:25:3, col:16>
// CHECK-NEXT: | |   | `-VarDecl [[ADDR_75:0x[a-z0-9]*]] <col:3, col:15> col:8 q 'S<T>' callinit
// CHECK-NEXT: | |   |   `-ParenListExpr [[ADDR_76:0x[a-z0-9]*]] <col:9, col:15> 'NULL TYPE'
// CHECK-NEXT: | |   |     |-IntegerLiteral [[ADDR_77:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | |   |     `-UnaryOperator [[ADDR_78:0x[a-z0-9]*]] <col:13, col:14> '<dependent type>' prefix '&' cannot overflow
// CHECK-NEXT: | |   |       `-DeclRefExpr [[ADDR_79:0x[a-z0-9]*]] <col:14> 'T' {{.*}}Var [[ADDR_73]] 't' 'T'
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_80:0x[a-z0-9]*]] <line:26:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_81:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_69]] <line:23:1, line:27:1> line:23:1 special[implementation={extension(allow_templates)}] 'int (S<int>)'
// CHECK-NEXT: |   |-TemplateArgument type 'int'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_9]] 'int'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_82:0x[a-z0-9]*]] <col:13, col:18> col:18 s 'S<int>':'S<int>'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_83:0x[a-z0-9]*]] <col:21, line:27:1>
// CHECK-NEXT: |     |-DeclStmt [[ADDR_84:0x[a-z0-9]*]] <line:24:3, col:6>
// CHECK-NEXT: |     | `-VarDecl [[ADDR_85:0x[a-z0-9]*]] <col:3, col:5> col:5 used t 'int':'int'
// CHECK-NEXT: |     |-DeclStmt [[ADDR_86:0x[a-z0-9]*]] <line:25:3, col:16>
// CHECK-NEXT: |     | `-VarDecl [[ADDR_87:0x[a-z0-9]*]] <col:3, col:15> col:8 q 'S<int>':'S<int>' callinit
// CHECK-NEXT: |     |   `-CXXConstructExpr [[ADDR_88:0x[a-z0-9]*]] <col:8, col:15> 'S<int>':'S<int>' 'void (int, int *)'
// CHECK-NEXT: |     |     |-IntegerLiteral [[ADDR_77]] <col:10> 'int' 0
// CHECK-NEXT: |     |     `-UnaryOperator [[ADDR_89:0x[a-z0-9]*]] <col:13, col:14> 'int *' prefix '&' cannot overflow
// CHECK-NEXT: |     |       `-DeclRefExpr [[ADDR_90:0x[a-z0-9]*]] <col:14> 'int':'int' {{.*}}Var [[ADDR_85]] 't' 'int':'int'
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_91:0x[a-z0-9]*]] <line:26:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_81]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_92:0x[a-z0-9]*]] <line:28:1, line:29:22> col:5 implicit also_after
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_93:0x[a-z0-9]*]] <line:28:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | `-FunctionDecl [[ADDR_94:0x[a-z0-9]*]] <line:29:1, col:22> col:5 also_after 'int (S<T>)'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_95:0x[a-z0-9]*]] <col:16, col:21> col:21 s 'S<T>'
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_96:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={extension(allow_templates)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_97:0x[a-z0-9]*]] <col:1> 'int (S<T>)' {{.*}}Function [[ADDR_98:0x[a-z0-9]*]] 'also_after[implementation={extension(allow_templates)}]' 'int (S<T>)'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_99:0x[a-z0-9]*]] <line:28:1, line:34:1> line:29:1 also_after[implementation={extension(allow_templates)}]
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_93]] <line:28:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | `-FunctionDecl [[ADDR_98]] <line:29:1, line:34:1> line:29:1 also_after[implementation={extension(allow_templates)}] 'int (S<T>)'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_95]] <col:16, col:21> col:21 s 'S<T>'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_100:0x[a-z0-9]*]] <col:24, line:34:1>
// CHECK-NEXT: |     |-DeclStmt [[ADDR_101:0x[a-z0-9]*]] <line:31:3, col:11>
// CHECK-NEXT: |     | `-VarDecl [[ADDR_102:0x[a-z0-9]*]] <col:3, col:10> col:10 referenced t 'double'
// CHECK-NEXT: |     |-DeclStmt [[ADDR_103:0x[a-z0-9]*]] <line:32:3, col:18>
// CHECK-NEXT: |     | `-VarDecl [[ADDR_104:0x[a-z0-9]*]] <col:3, col:17> col:8 q 'S<T>' callinit
// CHECK-NEXT: |     |   `-ParenListExpr [[ADDR_105:0x[a-z0-9]*]] <col:9, col:17> 'NULL TYPE'
// CHECK-NEXT: |     |     |-FloatingLiteral [[ADDR_106:0x[a-z0-9]*]] <col:10> 'double' 2.000000e+00
// CHECK-NEXT: |     |     `-UnaryOperator [[ADDR_107:0x[a-z0-9]*]] <col:15, col:16> 'double *' prefix '&' cannot overflow
// CHECK-NEXT: |     |       `-DeclRefExpr [[ADDR_108:0x[a-z0-9]*]] <col:16> 'double' {{.*}}Var [[ADDR_102]] 't' 'double'
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_109:0x[a-z0-9]*]] <line:33:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_110:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_111:0x[a-z0-9]*]] <line:37:1, line:40:1> line:38:5 also_after
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_112:0x[a-z0-9]*]] <line:37:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_113:0x[a-z0-9]*]] <line:38:1, line:40:1> line:38:5 also_after 'int (T)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_114:0x[a-z0-9]*]] <col:16, col:18> col:18 s 'T'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_115:0x[a-z0-9]*]] <col:21, line:40:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_116:0x[a-z0-9]*]] <line:39:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_117:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: | |-FunctionDecl [[ADDR_118:0x[a-z0-9]*]] <line:38:1, line:40:1> line:38:5 used also_after 'int (int)'
// CHECK-NEXT: | | |-TemplateArgument type 'int'
// CHECK-NEXT: | | | `-BuiltinType [[ADDR_9]] 'int'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_119:0x[a-z0-9]*]] <col:16, col:18> col:18 s 'int':'int'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_120:0x[a-z0-9]*]] <col:21, line:40:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_121:0x[a-z0-9]*]] <line:39:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_117]] <col:10> 'int' 0
// CHECK-NEXT: | `-FunctionDecl [[ADDR_122:0x[a-z0-9]*]] <line:38:1, line:40:1> line:38:5 used also_after 'int (double)'
// CHECK-NEXT: |   |-TemplateArgument type 'double'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_21]] 'double'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_123:0x[a-z0-9]*]] <col:16, col:18> col:18 s 'double':'double'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_124:0x[a-z0-9]*]] <col:21, line:40:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_125:0x[a-z0-9]*]] <line:39:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_117]] <col:10> 'int' 0
// CHECK-NEXT: `-FunctionDecl [[ADDR_126:0x[a-z0-9]*]] <line:42:1, line:45:1> line:42:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_127:0x[a-z0-9]*]] <col:12, line:45:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_128:0x[a-z0-9]*]] <line:44:3, col:98>
// CHECK-NEXT:       `-ExprWithCleanups [[ADDR_129:0x[a-z0-9]*]] <col:10, col:98> 'int'
// CHECK-NEXT:         `-BinaryOperator [[ADDR_130:0x[a-z0-9]*]] <col:10, col:98> 'int' '+'
// CHECK-NEXT:           |-BinaryOperator [[ADDR_131:0x[a-z0-9]*]] <col:10, col:74> 'int' '+'
// CHECK-NEXT:           | |-BinaryOperator [[ADDR_132:0x[a-z0-9]*]] <col:10, col:57> 'int' '+'
// CHECK-NEXT:           | | |-BinaryOperator [[ADDR_133:0x[a-z0-9]*]] <col:10, col:39> 'int' '+'
// CHECK-NEXT:           | | | |-CallExpr [[ADDR_134:0x[a-z0-9]*]] <col:10, col:23> 'int'
// CHECK-NEXT:           | | | | |-ImplicitCastExpr [[ADDR_135:0x[a-z0-9]*]] <col:10> 'int (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT:           | | | | | `-DeclRefExpr [[ADDR_136:0x[a-z0-9]*]] <col:10> 'int (int)' {{.*}}Function [[ADDR_32]] 'also_before' 'int (int)' (FunctionTemplate [[ADDR_22]] 'also_before')
// CHECK-NEXT:           | | | | `-IntegerLiteral [[ADDR_137:0x[a-z0-9]*]] <col:22> 'int' 0
// CHECK-NEXT:           | | | `-CallExpr [[ADDR_138:0x[a-z0-9]*]] <col:27, col:39> 'int'
// CHECK-NEXT:           | | |   |-ImplicitCastExpr [[ADDR_139:0x[a-z0-9]*]] <col:27> 'int (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT:           | | |   | `-DeclRefExpr [[ADDR_140:0x[a-z0-9]*]] <col:27> 'int (int)' {{.*}}Function [[ADDR_118]] 'also_after' 'int (int)' (FunctionTemplate [[ADDR_111]] 'also_after')
// CHECK-NEXT:           | | |   `-IntegerLiteral [[ADDR_141:0x[a-z0-9]*]] <col:38> 'int' 0
// CHECK-NEXT:           | | `-CallExpr [[ADDR_142:0x[a-z0-9]*]] <col:43, col:57> 'int'
// CHECK-NEXT:           | |   |-ImplicitCastExpr [[ADDR_143:0x[a-z0-9]*]] <col:43> 'int (*)(double)' <FunctionToPointerDecay>
// CHECK-NEXT:           | |   | `-DeclRefExpr [[ADDR_144:0x[a-z0-9]*]] <col:43> 'int (double)' {{.*}}Function [[ADDR_36]] 'also_before' 'int (double)' (FunctionTemplate [[ADDR_22]] 'also_before')
// CHECK-NEXT:           | |   `-FloatingLiteral [[ADDR_145:0x[a-z0-9]*]] <col:55> 'double' 0.000000e+00
// CHECK-NEXT:           | `-CallExpr [[ADDR_146:0x[a-z0-9]*]] <col:61, col:74> 'int'
// CHECK-NEXT:           |   |-ImplicitCastExpr [[ADDR_147:0x[a-z0-9]*]] <col:61> 'int (*)(double)' <FunctionToPointerDecay>
// CHECK-NEXT:           |   | `-DeclRefExpr [[ADDR_148:0x[a-z0-9]*]] <col:61> 'int (double)' {{.*}}Function [[ADDR_122]] 'also_after' 'int (double)' (FunctionTemplate [[ADDR_111]] 'also_after')
// CHECK-NEXT:           |   `-FloatingLiteral [[ADDR_149:0x[a-z0-9]*]] <col:72> 'double' 0.000000e+00
// CHECK-NEXT:           `-PseudoObjectExpr [[ADDR_150:0x[a-z0-9]*]] <col:78, col:98> 'int'
// CHECK-NEXT:             |-CallExpr [[ADDR_151:0x[a-z0-9]*]] <col:78, col:98> 'int'
// CHECK-NEXT:             | |-ImplicitCastExpr [[ADDR_152:0x[a-z0-9]*]] <col:78> 'int (*)(S<int>)' <FunctionToPointerDecay>
// CHECK-NEXT:             | | `-DeclRefExpr [[ADDR_153:0x[a-z0-9]*]] <col:78> 'int (S<int>)' {{.*}}Function [[ADDR_65]] 'special' 'int (S<int>)' (FunctionTemplate [[ADDR_58]] 'special')
// CHECK-NEXT:             | `-CXXConstructExpr [[ADDR_154:0x[a-z0-9]*]] <col:86, col:97> 'S<int>':'S<int>' 'void (S<int> &&) noexcept' elidable
// CHECK-NEXT:             |   `-MaterializeTemporaryExpr [[ADDR_155:0x[a-z0-9]*]] <col:86, col:97> 'S<int>':'S<int>' xvalue
// CHECK-NEXT:             |     `-CXXTemporaryObjectExpr [[ADDR_156:0x[a-z0-9]*]] <col:86, col:97> 'S<int>':'S<int>' 'void (int, int *)'
// CHECK-NEXT:             |       |-IntegerLiteral [[ADDR_157:0x[a-z0-9]*]] <col:93> 'int' 0
// CHECK-NEXT:             |       `-ImplicitCastExpr [[ADDR_158:0x[a-z0-9]*]] <col:96> 'int *' <NullToPointer>
// CHECK-NEXT:             |         `-IntegerLiteral [[ADDR_159:0x[a-z0-9]*]] <col:96> 'int' 0
// CHECK-NEXT:             `-CallExpr [[ADDR_160:0x[a-z0-9]*]] <line:23:1, line:44:98> 'int'
// CHECK-NEXT:               |-ImplicitCastExpr [[ADDR_161:0x[a-z0-9]*]] <line:23:1> 'int (*)(S<int>)' <FunctionToPointerDecay>
// CHECK-NEXT:               | `-DeclRefExpr [[ADDR_68]] <col:1> 'int (S<int>)' {{.*}}Function [[ADDR_69]] 'special[implementation={extension(allow_templates)}]' 'int (S<int>)'
// CHECK-NEXT:               `-CXXConstructExpr [[ADDR_162:0x[a-z0-9]*]] <line:44:86, col:97> 'S<int>':'S<int>' 'void (S<int> &&) noexcept' elidable
// CHECK-NEXT:                 `-MaterializeTemporaryExpr [[ADDR_163:0x[a-z0-9]*]] <col:86, col:97> 'S<int>':'S<int>' xvalue
// CHECK-NEXT:                   `-CXXTemporaryObjectExpr [[ADDR_156]] <col:86, col:97> 'S<int>':'S<int>' 'void (int, int *)'
// CHECK-NEXT:                     |-IntegerLiteral [[ADDR_157]] <col:93> 'int' 0
// CHECK-NEXT:                     `-ImplicitCastExpr [[ADDR_158]] <col:96> 'int *' <NullToPointer>
// CHECK-NEXT:                       `-IntegerLiteral [[ADDR_159]] <col:96> 'int' 0
