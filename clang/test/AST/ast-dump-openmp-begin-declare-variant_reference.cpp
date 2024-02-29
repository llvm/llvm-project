// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// expected-no-diagnostics

// Our very own std::move, copied from libcxx.
template <class _Tp> struct remove_reference { typedef _Tp type; };
template <class _Tp> struct remove_reference<_Tp &> { typedef _Tp type; };
template <class _Tp> struct remove_reference<_Tp &&> { typedef _Tp type; };

template <class _Tp>
inline typename remove_reference<_Tp>::type &&
move(_Tp &&__t) {
  typedef typename remove_reference<_Tp>::type _Up;
  return static_cast<_Up &&>(__t);
}
// ---

int Good, Bad;
int &also_before() {
  return Bad;
}
int also_before(float &&) {
  return 0;
}

#pragma omp begin declare variant match(implementation = {vendor(score(100) \
                                                                 : llvm)})
int also_after(void) {
  return 1;
}
int also_after(int &) {
  return 2;
}
// This one does overload the int(*)(double&) version!
int also_after(double &) {
  return 0;
}
int also_after(double &&) {
  return 3;
}
int also_after(short &) {
  return 5;
}
int also_after(short &&) {
  return 0;
}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation = {vendor(score(0) \
                                                                 : llvm)})
// This one does overload the int&(*)(void) version!
int &also_before() {
  return Good;
}
// This one does *not* overload the int(*)(float&&) version!
int also_before(float &) {
  return 6;
}
#pragma omp end declare variant

int also_after(void) {
  return 7;
}
int also_after(int) {
  return 8;
}
int also_after(double &) {
  return 9;
}
int also_after(short &&) {
  return 10;
}

int test1() {
  // Should return 0.
  double d;
  return also_after(d);
}

int test2() {
  // Should return 0.
  return &also_before() == &Good;
}

int test3(float &&f) {
  // Should return 0.
  return also_before(move(f));
}

int test4(short &&s) {
  // Should return 0.
  return also_after(move(s));
}

int test(float &&f, short &&s) {
  // Should return 0.
  return test1() + test2() + test3(move(f)) + test4(move(s));
}

// CHECK:      |-ClassTemplateDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, col:66> col:29 remove_reference
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_1:0x[a-z0-9]*]] <col:11, col:17> col:17 referenced class depth 0 index 0 _Tp
// CHECK-NEXT: | |-CXXRecordDecl [[ADDR_2:0x[a-z0-9]*]] <col:22, col:66> col:29 struct remove_reference definition
// CHECK-NEXT: | | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT: | | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT: | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: | | |-CXXRecordDecl [[ADDR_3:0x[a-z0-9]*]] <col:22, col:29> col:29 implicit struct remove_reference
// CHECK-NEXT: | | `-TypedefDecl [[ADDR_4:0x[a-z0-9]*]] <col:48, col:60> col:60 type '_Tp'
// CHECK-NEXT: | |   `-TemplateTypeParmType [[ADDR_5:0x[a-z0-9]*]] '_Tp' dependent depth 0 index 0
// CHECK-NEXT: | |     `-TemplateTypeParm [[ADDR_1]] '_Tp'
// CHECK-NEXT: | |-ClassTemplateSpecializationDecl [[ADDR_6:0x[a-z0-9]*]] <line:6:1, col:73> col:29 struct remove_reference definition
// CHECK-NEXT: | | |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT: | | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT: | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: | | |-TemplateArgument type 'float &'
// CHECK-NEXT: | | | `-LValueReferenceType [[ADDR_7:0x[a-z0-9]*]] 'float &'
// CHECK-NEXT: | | |   `-BuiltinType [[ADDR_8:0x[a-z0-9]*]] 'float'
// CHECK-NEXT: | | |-CXXRecordDecl [[ADDR_9:0x[a-z0-9]*]] <col:22, col:29> col:29 implicit struct remove_reference
// CHECK-NEXT: | | `-TypedefDecl [[ADDR_10:0x[a-z0-9]*]] <col:55, col:67> col:67 referenced type 'float'
// CHECK-NEXT: | |   `-SubstTemplateTypeParmType [[ADDR_11:0x[a-z0-9]*]] 'float' sugar class depth 0 index 0 _Tp
// CHECK-NEXT: | |     |-ClassTemplateSpecialization [[ADDR_6]] 'remove_reference'
// CHECK-NEXT: | |     `-BuiltinType [[ADDR_8]] 'float'
// CHECK-NEXT: | `-ClassTemplateSpecializationDecl [[ADDR_14:0x[a-z0-9]*]] <col:1, col:73> col:29 struct remove_reference definition
// CHECK-NEXT: |   |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: |   |-TemplateArgument type 'short &'
// CHECK-NEXT: |   | `-LValueReferenceType [[ADDR_15:0x[a-z0-9]*]] 'short &'
// CHECK-NEXT: |   |   `-BuiltinType [[ADDR_16:0x[a-z0-9]*]] 'short'
// CHECK-NEXT: |   |-CXXRecordDecl [[ADDR_17:0x[a-z0-9]*]] <col:22, col:29> col:29 implicit struct remove_reference
// CHECK-NEXT: |   `-TypedefDecl [[ADDR_18:0x[a-z0-9]*]] <col:55, col:67> col:67 referenced type 'short'
// CHECK-NEXT: |     `-SubstTemplateTypeParmType [[ADDR_19:0x[a-z0-9]*]] 'short' sugar class depth 0 index 0 _Tp
// CHECK-NEXT: |       |-ClassTemplateSpecialization [[ADDR_14]] 'remove_reference'
// CHECK-NEXT: |       `-BuiltinType [[ADDR_16]] 'short'
// CHECK-NEXT: |-ClassTemplatePartialSpecializationDecl [[ADDR_20:0x[a-z0-9]*]] <col:1, col:73> col:29 struct remove_reference definition
// CHECK-NEXT: | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT: | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: | |-TemplateArgument type 'type-parameter-0-0 &'
// CHECK-NEXT: | | `-LValueReferenceType [[ADDR_21:0x[a-z0-9]*]] 'type-parameter-0-0 &' dependent
// CHECK-NEXT: | |   `-TemplateTypeParmType [[ADDR_22:0x[a-z0-9]*]] 'type-parameter-0-0' dependent depth 0 index 0
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_13:0x[a-z0-9]*]] <col:11, col:17> col:17 referenced class depth 0 index 0 _Tp
// CHECK-NEXT: | |-CXXRecordDecl [[ADDR_23:0x[a-z0-9]*]] <col:22, col:29> col:29 implicit struct remove_reference
// CHECK-NEXT: | `-TypedefDecl [[ADDR_24:0x[a-z0-9]*]] <col:55, col:67> col:67 type '_Tp'
// CHECK-NEXT: |   `-TemplateTypeParmType [[ADDR_12:0x[a-z0-9]*]] '_Tp' dependent depth 0 index 0
// CHECK-NEXT: |     `-TemplateTypeParm [[ADDR_13]] '_Tp'
// CHECK-NEXT: |-ClassTemplatePartialSpecializationDecl [[ADDR_25:0x[a-z0-9]*]] <line:7:1, col:74> col:29 struct remove_reference definition
// CHECK-NEXT: | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT: | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT: | |-TemplateArgument type 'type-parameter-0-0 &&'
// CHECK-NEXT: | | `-RValueReferenceType [[ADDR_26:0x[a-z0-9]*]] 'type-parameter-0-0 &&' dependent
// CHECK-NEXT: | |   `-TemplateTypeParmType [[ADDR_22]] 'type-parameter-0-0' dependent depth 0 index 0
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_27:0x[a-z0-9]*]] <col:11, col:17> col:17 referenced class depth 0 index 0 _Tp
// CHECK-NEXT: | |-CXXRecordDecl [[ADDR_28:0x[a-z0-9]*]] <col:22, col:29> col:29 implicit struct remove_reference
// CHECK-NEXT: | `-TypedefDecl [[ADDR_29:0x[a-z0-9]*]] <col:56, col:68> col:68 type '_Tp'
// CHECK-NEXT: |   `-TemplateTypeParmType [[ADDR_30:0x[a-z0-9]*]] '_Tp' dependent depth 0 index 0
// CHECK-NEXT: |     `-TemplateTypeParm [[ADDR_27]] '_Tp'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_31:0x[a-z0-9]*]] <line:9:1, line:14:1> line:11:1 move
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_32:0x[a-z0-9]*]] <line:9:11, col:17> col:17 referenced class depth 0 index 0 _Tp
// CHECK-NEXT: | |-FunctionDecl [[ADDR_33:0x[a-z0-9]*]] <line:10:1, line:14:1> line:11:1 move 'typename remove_reference<_Tp>::type &&(_Tp &&)' inline
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_34:0x[a-z0-9]*]] <col:6, col:12> col:12 referenced __t '_Tp &&'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_35:0x[a-z0-9]*]] <col:17, line:14:1>
// CHECK-NEXT: | |   |-DeclStmt [[ADDR_36:0x[a-z0-9]*]] <line:12:3, col:51>
// CHECK-NEXT: | |   | `-TypedefDecl [[ADDR_37:0x[a-z0-9]*]] <col:3, col:48> col:48 referenced _Up 'typename remove_reference<_Tp>::type'
// CHECK-NEXT: | |   |   `-DependentNameType [[ADDR_38:0x[a-z0-9]*]] 'typename remove_reference<_Tp>::type' dependent
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_39:0x[a-z0-9]*]] <line:13:3, col:33>
// CHECK-NEXT: | |     `-CXXStaticCastExpr [[ADDR_40:0x[a-z0-9]*]] <col:10, col:33> '_Up':'typename remove_reference<_Tp>::type' xvalue static_cast<_Up &&> <Dependent>
// CHECK-NEXT: | |       `-DeclRefExpr [[ADDR_41:0x[a-z0-9]*]] <col:30> '_Tp' {{.*}}ParmVar [[ADDR_34]] '__t' '_Tp &&'
// CHECK-NEXT: | |-FunctionDecl [[ADDR_42:0x[a-z0-9]*]] <line:10:1, line:14:1> line:11:1 used move 'typename remove_reference<float &>::type &&(float &)' implicit_instantiation inline
// CHECK-NEXT: | | |-TemplateArgument type 'float &'
// CHECK-NEXT: | | | `-LValueReferenceType [[ADDR_7]] 'float &'
// CHECK-NEXT: | | |   `-BuiltinType [[ADDR_8]] 'float'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_43:0x[a-z0-9]*]] <col:6, col:12> col:12 used __t 'float &'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_44:0x[a-z0-9]*]] <col:17, line:14:1>
// CHECK-NEXT: | |   |-DeclStmt [[ADDR_45:0x[a-z0-9]*]] <line:12:3, col:51>
// CHECK-NEXT: | |   | `-TypedefDecl [[ADDR_46:0x[a-z0-9]*]] <col:3, col:48> col:48 referenced _Up 'typename remove_reference<float &>::type':'float'
// CHECK-NEXT: | |   |   `-ElaboratedType [[ADDR_47:0x[a-z0-9]*]] 'typename remove_reference<float &>::type' sugar
// CHECK-NEXT: | |   |     `-TypedefType [[ADDR_48:0x[a-z0-9]*]] 'remove_reference<float &>::type' sugar
// CHECK-NEXT: | |   |       |-Typedef [[ADDR_10]] 'type'
// CHECK-NEXT: | |   |       `-SubstTemplateTypeParmType [[ADDR_11]] 'float' sugar class depth 0 index 0 _Tp
// CHECK-NEXT: | |   |         |-ClassTemplateSpecialization [[ADDR_6]] 'remove_reference'
// CHECK-NEXT: | |   |         `-BuiltinType [[ADDR_8]] 'float'
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_49:0x[a-z0-9]*]] <line:13:3, col:33>
// CHECK-NEXT: | |     `-CXXStaticCastExpr [[ADDR_50:0x[a-z0-9]*]] <col:10, col:33> '_Up':'float' xvalue static_cast<_Up &&> <NoOp>
// CHECK-NEXT: | |       `-DeclRefExpr [[ADDR_51:0x[a-z0-9]*]] <col:30> 'float' {{.*}}ParmVar [[ADDR_43]] '__t' 'float &'
// CHECK-NEXT: | `-FunctionDecl [[ADDR_52:0x[a-z0-9]*]] <line:10:1, line:14:1> line:11:1 used move 'typename remove_reference<short &>::type &&(short &)' implicit_instantiation inline
// CHECK-NEXT: |   |-TemplateArgument type 'short &'
// CHECK-NEXT: |   | `-LValueReferenceType [[ADDR_15]] 'short &'
// CHECK-NEXT: |   |   `-BuiltinType [[ADDR_16]] 'short'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_53:0x[a-z0-9]*]] <col:6, col:12> col:12 used __t 'short &'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_54:0x[a-z0-9]*]] <col:17, line:14:1>
// CHECK-NEXT: |     |-DeclStmt [[ADDR_55:0x[a-z0-9]*]] <line:12:3, col:51>
// CHECK-NEXT: |     | `-TypedefDecl [[ADDR_56:0x[a-z0-9]*]] <col:3, col:48> col:48 referenced _Up 'typename remove_reference<short &>::type':'short'
// CHECK-NEXT: |     |   `-ElaboratedType [[ADDR_57:0x[a-z0-9]*]] 'typename remove_reference<short &>::type' sugar
// CHECK-NEXT: |     |     `-TypedefType [[ADDR_58:0x[a-z0-9]*]] 'remove_reference<short &>::type' sugar
// CHECK-NEXT: |     |       |-Typedef [[ADDR_18]] 'type'
// CHECK-NEXT: |     |       `-SubstTemplateTypeParmType [[ADDR_19]] 'short' sugar class depth 0 index 0 _Tp
// CHECK-NEXT: |     |         |-ClassTemplateSpecialization [[ADDR_14]] 'remove_reference'
// CHECK-NEXT: |     |         `-BuiltinType [[ADDR_16]] 'short'
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_59:0x[a-z0-9]*]] <line:13:3, col:33>
// CHECK-NEXT: |       `-CXXStaticCastExpr [[ADDR_60:0x[a-z0-9]*]] <col:10, col:33> '_Up':'short' xvalue static_cast<_Up &&> <NoOp>
// CHECK-NEXT: |         `-DeclRefExpr [[ADDR_61:0x[a-z0-9]*]] <col:30> 'short' {{.*}}ParmVar [[ADDR_53]] '__t' 'short &'
// CHECK-NEXT: |-VarDecl [[ADDR_62:0x[a-z0-9]*]] <line:17:1, col:5> col:5 used Good 'int'
// CHECK-NEXT: |-VarDecl [[ADDR_63:0x[a-z0-9]*]] <col:1, col:11> col:11 used Bad 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_64:0x[a-z0-9]*]] <line:18:1, line:20:1> line:18:6 used also_before 'int &({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_65:0x[a-z0-9]*]] <col:20, line:20:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_66:0x[a-z0-9]*]] <line:19:3, col:10>
// CHECK-NEXT: | |   `-DeclRefExpr [[ADDR_67:0x[a-z0-9]*]] <col:10> 'int' {{.*}}Var [[ADDR_63]] 'Bad' 'int'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_68:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(0): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_69:0x[a-z0-9]*]] <line:50:1> 'int &({{.*}})' {{.*}}Function [[ADDR_70:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int &({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_71:0x[a-z0-9]*]] <line:21:1, line:23:1> line:21:5 used also_before 'int (float &&)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_72:0x[a-z0-9]*]] <col:17, col:23> col:25 'float &&'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_73:0x[a-z0-9]*]] <col:27, line:23:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_74:0x[a-z0-9]*]] <line:22:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_75:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_76:0x[a-z0-9]*]] <line:27:1, col:20> col:5 implicit also_after 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_77:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_78:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_79:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_79]] <col:1, line:29:1> line:27:1 also_after[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_80:0x[a-z0-9]*]] <col:22, line:29:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_81:0x[a-z0-9]*]] <line:28:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_82:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |-FunctionDecl [[ADDR_83:0x[a-z0-9]*]] <line:30:1, col:21> col:5 implicit also_after 'int (int &)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_84:0x[a-z0-9]*]] <col:16, col:20> col:21 'int &'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_85:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_86:0x[a-z0-9]*]] <col:1> 'int (int &)' {{.*}}Function [[ADDR_87:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int (int &)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_87]] <col:1, line:32:1> line:30:1 also_after[implementation={vendor(llvm)}] 'int (int &)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_84]] <col:16, col:20> col:21 'int &'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_88:0x[a-z0-9]*]] <col:23, line:32:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_89:0x[a-z0-9]*]] <line:31:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_90:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: |-FunctionDecl [[ADDR_91:0x[a-z0-9]*]] <line:34:1, col:24> col:5 implicit used also_after 'int (double &)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_92:0x[a-z0-9]*]] <col:16, col:23> col:24 'double &'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_93:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_94:0x[a-z0-9]*]] <col:1> 'int (double &)' {{.*}}Function [[ADDR_95:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int (double &)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_95]] <col:1, line:36:1> line:34:1 also_after[implementation={vendor(llvm)}] 'int (double &)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_92]] <col:16, col:23> col:24 'double &'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_96:0x[a-z0-9]*]] <col:26, line:36:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_97:0x[a-z0-9]*]] <line:35:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_98:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_99:0x[a-z0-9]*]] <line:37:1, col:25> col:5 implicit also_after 'int (double &&)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_100:0x[a-z0-9]*]] <col:16, col:23> col:25 'double &&'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_101:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_102:0x[a-z0-9]*]] <col:1> 'int (double &&)' {{.*}}Function [[ADDR_103:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int (double &&)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_103]] <col:1, line:39:1> line:37:1 also_after[implementation={vendor(llvm)}] 'int (double &&)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_100]] <col:16, col:23> col:25 'double &&'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_104:0x[a-z0-9]*]] <col:27, line:39:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_105:0x[a-z0-9]*]] <line:38:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_106:0x[a-z0-9]*]] <col:10> 'int' 3
// CHECK-NEXT: |-FunctionDecl [[ADDR_107:0x[a-z0-9]*]] <line:40:1, col:23> col:5 implicit also_after 'int (short &)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_108:0x[a-z0-9]*]] <col:16, col:22> col:23 'short &'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_109:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_110:0x[a-z0-9]*]] <col:1> 'int (short &)' {{.*}}Function [[ADDR_111:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int (short &)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_111]] <col:1, line:42:1> line:40:1 also_after[implementation={vendor(llvm)}] 'int (short &)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_108]] <col:16, col:22> col:23 'short &'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_112:0x[a-z0-9]*]] <col:25, line:42:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_113:0x[a-z0-9]*]] <line:41:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_114:0x[a-z0-9]*]] <col:10> 'int' 5
// CHECK-NEXT: |-FunctionDecl [[ADDR_115:0x[a-z0-9]*]] <line:43:1, col:24> col:5 implicit used also_after 'int (short &&)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_116:0x[a-z0-9]*]] <col:16, col:22> col:24 'short &&'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_117:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_118:0x[a-z0-9]*]] <col:1> 'int (short &&)' {{.*}}Function [[ADDR_119:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int (short &&)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_119]] <col:1, line:45:1> line:43:1 also_after[implementation={vendor(llvm)}] 'int (short &&)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_116]] <col:16, col:22> col:24 'short &&'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_120:0x[a-z0-9]*]] <col:26, line:45:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_121:0x[a-z0-9]*]] <line:44:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_122:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_70]] <line:50:1, line:52:1> line:50:1 also_before[implementation={vendor(llvm)}] 'int &({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_123:0x[a-z0-9]*]] <col:20, line:52:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_124:0x[a-z0-9]*]] <line:51:3, col:10>
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_125:0x[a-z0-9]*]] <col:10> 'int' {{.*}}Var [[ADDR_62]] 'Good' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_126:0x[a-z0-9]*]] <line:54:1, col:24> col:5 implicit also_before 'int (float &)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_127:0x[a-z0-9]*]] <col:17, col:23> col:24 'float &'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_128:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(score(0): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_129:0x[a-z0-9]*]] <col:1> 'int (float &)' {{.*}}Function [[ADDR_130:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int (float &)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_130]] <col:1, line:56:1> line:54:1 also_before[implementation={vendor(llvm)}] 'int (float &)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_127]] <col:17, col:23> col:24 'float &'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_131:0x[a-z0-9]*]] <col:26, line:56:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_132:0x[a-z0-9]*]] <line:55:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_133:0x[a-z0-9]*]] <col:10> 'int' 6
// CHECK-NEXT: |-FunctionDecl [[ADDR_134:0x[a-z0-9]*]] prev [[ADDR_76]] <line:59:1, line:61:1> line:59:5 also_after 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_135:0x[a-z0-9]*]] <col:22, line:61:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_136:0x[a-z0-9]*]] <line:60:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_137:0x[a-z0-9]*]] <col:10> 'int' 7
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_138:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_78]] <line:27:1> 'int ({{.*}})' {{.*}}Function [[ADDR_79]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_139:0x[a-z0-9]*]] <line:62:1, line:64:1> line:62:5 also_after 'int (int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_140:0x[a-z0-9]*]] <col:16> col:19 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_141:0x[a-z0-9]*]] <col:21, line:64:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_142:0x[a-z0-9]*]] <line:63:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_143:0x[a-z0-9]*]] <col:10> 'int' 8
// CHECK-NEXT: |-FunctionDecl [[ADDR_144:0x[a-z0-9]*]] prev [[ADDR_91]] <line:65:1, line:67:1> line:65:5 used also_after 'int (double &)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_145:0x[a-z0-9]*]] <col:16, col:23> col:24 'double &'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_146:0x[a-z0-9]*]] <col:26, line:67:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_147:0x[a-z0-9]*]] <line:66:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_148:0x[a-z0-9]*]] <col:10> 'int' 9
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_149:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_94]] <line:34:1> 'int (double &)' {{.*}}Function [[ADDR_95]] 'also_after[implementation={vendor(llvm)}]' 'int (double &)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_150:0x[a-z0-9]*]] prev [[ADDR_115]] <line:68:1, line:70:1> line:68:5 used also_after 'int (short &&)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_151:0x[a-z0-9]*]] <col:16, col:22> col:24 'short &&'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_152:0x[a-z0-9]*]] <col:26, line:70:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_153:0x[a-z0-9]*]] <line:69:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_154:0x[a-z0-9]*]] <col:10> 'int' 10
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_155:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(score(100): llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_118]] <line:43:1> 'int (short &&)' {{.*}}Function [[ADDR_119]] 'also_after[implementation={vendor(llvm)}]' 'int (short &&)'
// CHECK-NEXT: |-FunctionDecl [[ADDR_156:0x[a-z0-9]*]] <line:72:1, line:76:1> line:72:5 used test1 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_157:0x[a-z0-9]*]] <col:13, line:76:1>
// CHECK-NEXT: |   |-DeclStmt [[ADDR_158:0x[a-z0-9]*]] <line:74:3, col:11>
// CHECK-NEXT: |   | `-VarDecl [[ADDR_159:0x[a-z0-9]*]] <col:3, col:10> col:10 used d 'double'
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_160:0x[a-z0-9]*]] <line:75:3, col:22>
// CHECK-NEXT: |     `-PseudoObjectExpr [[ADDR_161:0x[a-z0-9]*]] <col:10, col:22> 'int'
// CHECK-NEXT: |       |-CallExpr [[ADDR_162:0x[a-z0-9]*]] <col:10, col:22> 'int'
// CHECK-NEXT: |       | |-ImplicitCastExpr [[ADDR_163:0x[a-z0-9]*]] <col:10> 'int (*)(double &)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_164:0x[a-z0-9]*]] <col:10> 'int (double &)' {{.*}}Function [[ADDR_144]] 'also_after' 'int (double &)'
// CHECK-NEXT: |       | `-DeclRefExpr [[ADDR_165:0x[a-z0-9]*]] <col:21> 'double' {{.*}}Var [[ADDR_159]] 'd' 'double'
// CHECK-NEXT: |       `-CallExpr [[ADDR_166:0x[a-z0-9]*]] <line:34:1, line:75:22> 'int'
// CHECK-NEXT: |         |-ImplicitCastExpr [[ADDR_167:0x[a-z0-9]*]] <line:34:1> 'int (*)(double &)' <FunctionToPointerDecay>
// CHECK-NEXT: |         | `-DeclRefExpr [[ADDR_94]] <col:1> 'int (double &)' {{.*}}Function [[ADDR_95]] 'also_after[implementation={vendor(llvm)}]' 'int (double &)'
// CHECK-NEXT: |         `-DeclRefExpr [[ADDR_165]] <line:75:21> 'double' {{.*}}Var [[ADDR_159]] 'd' 'double'
// CHECK-NEXT: |-FunctionDecl [[ADDR_168:0x[a-z0-9]*]] <line:78:1, line:81:1> line:78:5 used test2 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_169:0x[a-z0-9]*]] <col:13, line:81:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_170:0x[a-z0-9]*]] <line:80:3, col:29>
// CHECK-NEXT: |     `-ImplicitCastExpr [[ADDR_171:0x[a-z0-9]*]] <col:10, col:29> 'int' <IntegralCast>
// CHECK-NEXT: |       `-BinaryOperator [[ADDR_172:0x[a-z0-9]*]] <col:10, col:29> 'bool' '=='
// CHECK-NEXT: |         |-UnaryOperator [[ADDR_173:0x[a-z0-9]*]] <col:10, col:23> 'int *' prefix '&' cannot overflow
// CHECK-NEXT: |         | `-PseudoObjectExpr [[ADDR_174:0x[a-z0-9]*]] <col:11, col:23> 'int' lvalue
// CHECK-NEXT: |         |   |-CallExpr [[ADDR_175:0x[a-z0-9]*]] <col:11, col:23> 'int' lvalue
// CHECK-NEXT: |         |   | `-ImplicitCastExpr [[ADDR_176:0x[a-z0-9]*]] <col:11> 'int &(*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |         |   |   `-DeclRefExpr [[ADDR_177:0x[a-z0-9]*]] <col:11> 'int &({{.*}})' {{.*}}Function [[ADDR_64]] 'also_before' 'int &({{.*}})'
// CHECK-NEXT: |         |   `-CallExpr [[ADDR_178:0x[a-z0-9]*]] <line:50:1, line:80:23> 'int' lvalue
// CHECK-NEXT: |         |     `-ImplicitCastExpr [[ADDR_179:0x[a-z0-9]*]] <line:50:1> 'int &(*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |         |       `-DeclRefExpr [[ADDR_69]] <col:1> 'int &({{.*}})' {{.*}}Function [[ADDR_70]] 'also_before[implementation={vendor(llvm)}]' 'int &({{.*}})'
// CHECK-NEXT: |         `-UnaryOperator [[ADDR_180:0x[a-z0-9]*]] <line:80:28, col:29> 'int *' prefix '&' cannot overflow
// CHECK-NEXT: |           `-DeclRefExpr [[ADDR_181:0x[a-z0-9]*]] <col:29> 'int' {{.*}}Var [[ADDR_62]] 'Good' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_182:0x[a-z0-9]*]] <line:83:1, line:86:1> line:83:5 used test3 'int (float &&)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_183:0x[a-z0-9]*]] <col:11, col:19> col:19 used f 'float &&'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_184:0x[a-z0-9]*]] <col:22, line:86:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_185:0x[a-z0-9]*]] <line:85:3, col:29>
// CHECK-NEXT: |     `-CallExpr [[ADDR_186:0x[a-z0-9]*]] <col:10, col:29> 'int'
// CHECK-NEXT: |       |-ImplicitCastExpr [[ADDR_187:0x[a-z0-9]*]] <col:10> 'int (*)(float &&)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | `-DeclRefExpr [[ADDR_188:0x[a-z0-9]*]] <col:10> 'int (float &&)' {{.*}}Function [[ADDR_71]] 'also_before' 'int (float &&)'
// CHECK-NEXT: |       `-CallExpr [[ADDR_189:0x[a-z0-9]*]] <col:22, col:28> 'typename remove_reference<float &>::type':'float' xvalue
// CHECK-NEXT: |         |-ImplicitCastExpr [[ADDR_190:0x[a-z0-9]*]] <col:22> 'typename remove_reference<float &>::type &&(*)(float &)' <FunctionToPointerDecay>
// CHECK-NEXT: |         | `-DeclRefExpr [[ADDR_191:0x[a-z0-9]*]] <col:22> 'typename remove_reference<float &>::type &&(float &)' {{.*}}Function [[ADDR_42]] 'move' 'typename remove_reference<float &>::type &&(float &)' (FunctionTemplate [[ADDR_31]] 'move')
// CHECK-NEXT: |         `-DeclRefExpr [[ADDR_192:0x[a-z0-9]*]] <col:27> 'float' {{.*}}ParmVar [[ADDR_183]] 'f' 'float &&'
// CHECK-NEXT: |-FunctionDecl [[ADDR_193:0x[a-z0-9]*]] <line:88:1, line:91:1> line:88:5 used test4 'int (short &&)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_194:0x[a-z0-9]*]] <col:11, col:19> col:19 used s 'short &&'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_195:0x[a-z0-9]*]] <col:22, line:91:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_196:0x[a-z0-9]*]] <line:90:3, col:28>
// CHECK-NEXT: |     `-PseudoObjectExpr [[ADDR_197:0x[a-z0-9]*]] <col:10, col:28> 'int'
// CHECK-NEXT: |       |-CallExpr [[ADDR_198:0x[a-z0-9]*]] <col:10, col:28> 'int'
// CHECK-NEXT: |       | |-ImplicitCastExpr [[ADDR_199:0x[a-z0-9]*]] <col:10> 'int (*)(short &&)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_200:0x[a-z0-9]*]] <col:10> 'int (short &&)' {{.*}}Function [[ADDR_150]] 'also_after' 'int (short &&)'
// CHECK-NEXT: |       | `-CallExpr [[ADDR_201:0x[a-z0-9]*]] <col:21, col:27> 'typename remove_reference<short &>::type':'short' xvalue
// CHECK-NEXT: |       |   |-ImplicitCastExpr [[ADDR_202:0x[a-z0-9]*]] <col:21> 'typename remove_reference<short &>::type &&(*)(short &)' <FunctionToPointerDecay>
// CHECK-NEXT: |       |   | `-DeclRefExpr [[ADDR_203:0x[a-z0-9]*]] <col:21> 'typename remove_reference<short &>::type &&(short &)' {{.*}}Function [[ADDR_52]] 'move' 'typename remove_reference<short &>::type &&(short &)' (FunctionTemplate [[ADDR_31]] 'move')
// CHECK-NEXT: |       |   `-DeclRefExpr [[ADDR_204:0x[a-z0-9]*]] <col:26> 'short' {{.*}}ParmVar [[ADDR_194]] 's' 'short &&'
// CHECK-NEXT: |       `-CallExpr [[ADDR_205:0x[a-z0-9]*]] <line:43:1, line:90:28> 'int'
// CHECK-NEXT: |         |-ImplicitCastExpr [[ADDR_206:0x[a-z0-9]*]] <line:43:1> 'int (*)(short &&)' <FunctionToPointerDecay>
// CHECK-NEXT: |         | `-DeclRefExpr [[ADDR_118]] <col:1> 'int (short &&)' {{.*}}Function [[ADDR_119]] 'also_after[implementation={vendor(llvm)}]' 'int (short &&)'
// CHECK-NEXT: |         `-CallExpr [[ADDR_201]] <line:90:21, col:27> 'typename remove_reference<short &>::type':'short' xvalue
// CHECK-NEXT: |           |-ImplicitCastExpr [[ADDR_202]] <col:21> 'typename remove_reference<short &>::type &&(*)(short &)' <FunctionToPointerDecay>
// CHECK-NEXT: |           | `-DeclRefExpr [[ADDR_203]] <col:21> 'typename remove_reference<short &>::type &&(short &)' {{.*}}Function [[ADDR_52]] 'move' 'typename remove_reference<short &>::type &&(short &)' (FunctionTemplate [[ADDR_31]] 'move')
// CHECK-NEXT: |           `-DeclRefExpr [[ADDR_204]] <col:26> 'short' {{.*}}ParmVar [[ADDR_194]] 's' 'short &&'
// CHECK-NEXT: `-FunctionDecl [[ADDR_207:0x[a-z0-9]*]] <line:93:1, line:96:1> line:93:5 test 'int (float &&, short &&)'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_208:0x[a-z0-9]*]] <col:10, col:18> col:18 used f 'float &&'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_209:0x[a-z0-9]*]] <col:21, col:29> col:29 used s 'short &&'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_210:0x[a-z0-9]*]] <col:32, line:96:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_211:0x[a-z0-9]*]] <line:95:3, col:60>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_212:0x[a-z0-9]*]] <col:10, col:60> 'int' '+'
// CHECK-NEXT:         |-BinaryOperator [[ADDR_213:0x[a-z0-9]*]] <col:10, col:43> 'int' '+'
// CHECK-NEXT:         | |-BinaryOperator [[ADDR_214:0x[a-z0-9]*]] <col:10, col:26> 'int' '+'
// CHECK-NEXT:         | | |-CallExpr [[ADDR_215:0x[a-z0-9]*]] <col:10, col:16> 'int'
// CHECK-NEXT:         | | | `-ImplicitCastExpr [[ADDR_216:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | | |   `-DeclRefExpr [[ADDR_217:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_156]] 'test1' 'int ({{.*}})'
// CHECK-NEXT:         | | `-CallExpr [[ADDR_218:0x[a-z0-9]*]] <col:20, col:26> 'int'
// CHECK-NEXT:         | |   `-ImplicitCastExpr [[ADDR_219:0x[a-z0-9]*]] <col:20> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | |     `-DeclRefExpr [[ADDR_220:0x[a-z0-9]*]] <col:20> 'int ({{.*}})' {{.*}}Function [[ADDR_168]] 'test2' 'int ({{.*}})'
// CHECK-NEXT:         | `-CallExpr [[ADDR_221:0x[a-z0-9]*]] <col:30, col:43> 'int'
// CHECK-NEXT:         |   |-ImplicitCastExpr [[ADDR_222:0x[a-z0-9]*]] <col:30> 'int (*)(float &&)' <FunctionToPointerDecay>
// CHECK-NEXT:         |   | `-DeclRefExpr [[ADDR_223:0x[a-z0-9]*]] <col:30> 'int (float &&)' {{.*}}Function [[ADDR_182]] 'test3' 'int (float &&)'
// CHECK-NEXT:         |   `-CallExpr [[ADDR_224:0x[a-z0-9]*]] <col:36, col:42> 'typename remove_reference<float &>::type':'float' xvalue
// CHECK-NEXT:         |     |-ImplicitCastExpr [[ADDR_225:0x[a-z0-9]*]] <col:36> 'typename remove_reference<float &>::type &&(*)(float &)' <FunctionToPointerDecay>
// CHECK-NEXT:         |     | `-DeclRefExpr [[ADDR_226:0x[a-z0-9]*]] <col:36> 'typename remove_reference<float &>::type &&(float &)' {{.*}}Function [[ADDR_42]] 'move' 'typename remove_reference<float &>::type &&(float &)' (FunctionTemplate [[ADDR_31]] 'move')
// CHECK-NEXT:         |     `-DeclRefExpr [[ADDR_227:0x[a-z0-9]*]] <col:41> 'float' {{.*}}ParmVar [[ADDR_208]] 'f' 'float &&'
// CHECK-NEXT:         `-CallExpr [[ADDR_228:0x[a-z0-9]*]] <col:47, col:60> 'int'
// CHECK-NEXT:           |-ImplicitCastExpr [[ADDR_229:0x[a-z0-9]*]] <col:47> 'int (*)(short &&)' <FunctionToPointerDecay>
// CHECK-NEXT:           | `-DeclRefExpr [[ADDR_230:0x[a-z0-9]*]] <col:47> 'int (short &&)' {{.*}}Function [[ADDR_193]] 'test4' 'int (short &&)'
// CHECK-NEXT:           `-CallExpr [[ADDR_231:0x[a-z0-9]*]] <col:53, col:59> 'typename remove_reference<short &>::type':'short' xvalue
// CHECK-NEXT:             |-ImplicitCastExpr [[ADDR_232:0x[a-z0-9]*]] <col:53> 'typename remove_reference<short &>::type &&(*)(short &)' <FunctionToPointerDecay>
// CHECK-NEXT:             | `-DeclRefExpr [[ADDR_233:0x[a-z0-9]*]] <col:53> 'typename remove_reference<short &>::type &&(short &)' {{.*}}Function [[ADDR_52]] 'move' 'typename remove_reference<short &>::type &&(short &)' (FunctionTemplate [[ADDR_31]] 'move')
// CHECK-NEXT:             `-DeclRefExpr [[ADDR_234:0x[a-z0-9]*]] <col:58> 'short' {{.*}}ParmVar [[ADDR_209]] 's' 'short &&'
