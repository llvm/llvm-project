// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s %std_cxx17- | FileCheck %s --check-prefixes=CHECK
// expected-no-diagnostics

template <class T>
struct Redeclared {
  void function() {}
};

template <class T>
struct Redeclared;

Redeclared<int> instantiation;

template <typename T>
void redeclaredFunction(T t) {
  (void)t;
}

template <typename T>
void redeclaredFunction(T t);

void instantiate() {
  redeclaredFunction(0);
}


// CHECK:      |-ClassTemplateDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}:4:1, line:7:1> line:5:8 Redeclared
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_1:0x[a-z0-9]*]] <line:4:11, col:17> col:17 class depth 0 index 0 T
// CHECK-NEXT: | |-CXXRecordDecl [[ADDR_2:0x[a-z0-9]*]] <line:5:1, line:7:1> line:5:8 struct Redeclared definition
// CHECK-NEXT: | | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT: | | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT: | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: | | | `-Destructor simple irrelevant trivial {{(constexpr )?}}needs_implicit
// CHECK-NEXT: | | |-CXXRecordDecl [[ADDR_3:0x[a-z0-9]*]] <col:1, col:8> col:8 implicit struct Redeclared
// CHECK-NEXT: | | `-CXXMethodDecl [[ADDR_4:0x[a-z0-9]*]] <line:6:3, col:20> col:8 function 'void ()' implicit-inline
// CHECK-NEXT: | |   `-CompoundStmt [[ADDR_5:0x[a-z0-9]*]] <col:19, col:20>
// CHECK-NEXT: | `-ClassTemplateSpecializationDecl [[ADDR_6:0x[a-z0-9]*]] <line:5:1, line:7:1> line:5:8 struct Redeclared definition implicit_instantiation
// CHECK-NEXT: |   |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT: |   | |-DefaultConstructor exists trivial constexpr defaulted_is_constexpr
// CHECK-NEXT: |   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT: |   | |-MoveConstructor exists simple trivial
// CHECK-NEXT: |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT: |   | `-Destructor simple irrelevant trivial constexpr needs_implicit
// CHECK-NEXT: |   |-TemplateArgument type 'int'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_7:0x[a-z0-9]*]] 'int'
// CHECK-NEXT: |   |-CXXRecordDecl [[ADDR_8:0x[a-z0-9]*]] <col:1, col:8> col:8 implicit struct Redeclared
// CHECK-NEXT: |   |-CXXMethodDecl [[ADDR_9:0x[a-z0-9]*]] <line:6:3, col:20> col:8 function 'void ()' implicit_instantiation implicit-inline instantiated_from [[ADDR_4]]
// CHECK-NEXT: |   |-CXXConstructorDecl [[ADDR_10:0x[a-z0-9]*]] <line:5:8> col:8 implicit used constexpr Redeclared 'void () noexcept' inline default trivial
// CHECK-NEXT: |   | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:8>
// CHECK-NEXT: |   |-CXXConstructorDecl [[ADDR_12:0x[a-z0-9]*]] <col:8> col:8 implicit constexpr Redeclared 'void (const Redeclared<int> &)' inline default trivial noexcept-unevaluated [[ADDR_12]]
// CHECK-NEXT: |   | `-ParmVarDecl [[ADDR_13:0x[a-z0-9]*]] <col:8> col:8 'const Redeclared<int> &'
// CHECK-NEXT: |   `-CXXConstructorDecl [[ADDR_14:0x[a-z0-9]*]] <col:8> col:8 implicit constexpr Redeclared 'void (Redeclared<int> &&)' inline default trivial noexcept-unevaluated [[ADDR_14]]
// CHECK-NEXT: |     `-ParmVarDecl [[ADDR_15:0x[a-z0-9]*]] <col:8> col:8 'Redeclared<int> &&'
// CHECK-NEXT: |-ClassTemplateDecl [[ADDR_16:0x[a-z0-9]*]] prev [[ADDR_0]] <line:9:1, line:10:8> col:8 Redeclared
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_17:0x[a-z0-9]*]] <line:9:11, col:17> col:17 class depth 0 index 0 T
// CHECK-NEXT: | |-CXXRecordDecl [[ADDR_18:0x[a-z0-9]*]] prev [[ADDR_2]] <line:10:1, col:8> col:8 struct Redeclared
// CHECK-NEXT: | `-ClassTemplateSpecialization [[ADDR_19:0x[a-z0-9]*]] 'Redeclared'
// CHECK-NEXT: |-VarDecl [[ADDR_20:0x[a-z0-9]*]] <line:12:1, col:17> col:17 instantiation 'Redeclared<int>' callinit
// CHECK-NEXT: | `-CXXConstructExpr [[ADDR_21:0x[a-z0-9]*]] <col:17> 'Redeclared<int>' 'void () noexcept'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_22:0x[a-z0-9]*]] <line:14:1, line:17:1> line:15:6 redeclaredFunction
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_23:0x[a-z0-9]*]] <line:14:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_24:0x[a-z0-9]*]] <line:15:1, line:17:1> line:15:6 redeclaredFunction 'void (T)'
// CHECK-NEXT: | | |-ParmVarDecl [[ADDR_25:0x[a-z0-9]*]] <col:25, col:27> col:27 referenced t 'T'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_26:0x[a-z0-9]*]] <col:30, line:17:1>
// CHECK-NEXT: | |   `-CStyleCastExpr [[ADDR_27:0x[a-z0-9]*]] <line:16:3, col:9> 'void' <ToVoid>
// CHECK-NEXT: | |     `-DeclRefExpr [[ADDR_28:0x[a-z0-9]*]] <col:9> 'T' lvalue ParmVar [[ADDR_25]] 't' 'T'
// CHECK-NEXT: | `-FunctionDecl [[ADDR_29:0x[a-z0-9]*]] <line:15:1, line:17:1> line:15:6 used redeclaredFunction 'void (int)' implicit_instantiation
// CHECK-NEXT: |   |-TemplateArgument type 'int'
// CHECK-NEXT: |   | `-BuiltinType [[ADDR_30:0x[a-z0-9]*]] 'int'
// CHECK-NEXT: |   |-ParmVarDecl [[ADDR_31:0x[a-z0-9]*]] <line:20:25, col:27> col:27 used t 'int'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_32:0x[a-z0-9]*]] <line:15:30, line:17:1>
// CHECK-NEXT: |     `-CStyleCastExpr [[ADDR_33:0x[a-z0-9]*]] <line:16:3, col:9> 'void' <ToVoid>
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_34:0x[a-z0-9]*]] <col:9> 'int' lvalue ParmVar [[ADDR_31]] 't' 'int'
// CHECK-NEXT: |-FunctionTemplateDecl [[ADDR_35:0x[a-z0-9]*]] prev [[ADDR_22]] <line:19:1, line:20:28> col:6 redeclaredFunction
// CHECK-NEXT: | |-TemplateTypeParmDecl [[ADDR_36:0x[a-z0-9]*]] <line:19:11, col:20> col:20 referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl [[ADDR_37:0x[a-z0-9]*]] prev [[ADDR_24]] <line:20:1, col:28> col:6 redeclaredFunction 'void (T)'
// CHECK-NEXT: | | `-ParmVarDecl [[ADDR_38:0x[a-z0-9]*]] <col:25, col:27> col:27 t 'T'
// CHECK-NEXT: | `-Function [[ADDR_39:0x[a-z0-9]*]] 'redeclaredFunction' 'void (int)'
// CHECK-NEXT: `-FunctionDecl [[ADDR_40:0x[a-z0-9]*]] <line:22:1, line:24:1> line:22:6 instantiate 'void ()'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_41:0x[a-z0-9]*]] <col:20, line:24:1>
// CHECK-NEXT:     `-CallExpr [[ADDR_42:0x[a-z0-9]*]] <line:23:3, col:23> 'void'
// CHECK-NEXT:       |-ImplicitCastExpr [[ADDR_43:0x[a-z0-9]*]] <col:3> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT:       | `-DeclRefExpr [[ADDR_44:0x[a-z0-9]*]] <col:3> 'void (int)' lvalue Function [[ADDR_29]] 'redeclaredFunction' 'void (int)' (FunctionTemplate [[ADDR_35]] 'redeclaredFunction')
// CHECK-NEXT:       `-IntegerLiteral [[ADDR_45:0x[a-z0-9]*]] <col:22> 'int' 0
