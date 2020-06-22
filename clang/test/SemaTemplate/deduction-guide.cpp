// RUN: %clang_cc1 -std=c++2a -verify -ast-dump -ast-dump-decl-types -ast-dump-filter "deduction guide" %s | FileCheck %s

template<auto ...> struct X {};

template<typename T, typename ...Ts> struct A { // expected-note 2{{candidate}}
  template<Ts ...Ns, T *...Ps> A(X<Ps...>, Ts (*...qs)[Ns]); // expected-note {{candidate}}
};
int arr1[3], arr2[3];
short arr3[4];
// FIXME: The CTAD deduction here succeeds, but the initialization deduction spuriously fails.
A a(X<&arr1, &arr2>{}, &arr1, &arr2, &arr3); // FIXME: expected-error {{no matching constructor}}
using AT = decltype(a);
using AT = A<int[3], int, int, short>;

// CHECK: Dumping <deduction guide for A>:
// CHECK: FunctionTemplateDecl
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 1 ... Ts
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'Ts...' depth 0 index 2 ... Ns
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'T *' depth 0 index 3 ... Ps
// CHECK: |-CXXDeductionGuideDecl
// CHECK: | |-ParmVarDecl {{.*}} 'X<Ps...>'
// CHECK: | `-ParmVarDecl {{.*}} 'Ts (*)[Ns]...' pack
// CHECK: `-CXXDeductionGuideDecl
// CHECK:   |-TemplateArgument type 'int [3]'
// CHECK:   |-TemplateArgument pack
// CHECK:   | |-TemplateArgument type 'int'
// CHECK:   | |-TemplateArgument type 'int'
// CHECK:   | `-TemplateArgument type 'short'
// CHECK:   |-TemplateArgument pack
// CHECK:   | |-TemplateArgument integral 3
// CHECK:   | |-TemplateArgument integral 3
// CHECK:   | `-TemplateArgument integral 4
// CHECK:   |-TemplateArgument pack
// CHECK:   | |-TemplateArgument decl
// CHECK:   | | `-Var {{.*}} 'arr1' 'int [3]'
// CHECK:   | `-TemplateArgument decl
// CHECK:   |   `-Var {{.*}} 'arr2' 'int [3]'
// CHECK:   |-ParmVarDecl {{.*}} 'X<&arr1, &arr2>':'X<&arr1, &arr2>'
// CHECK:   |-ParmVarDecl {{.*}} 'int (*)[3]'
// CHECK:   |-ParmVarDecl {{.*}} 'int (*)[3]'
// CHECK:   `-ParmVarDecl {{.*}} 'short (*)[4]'
// CHECK: FunctionProtoType {{.*}} 'auto (X<Ps...>, Ts (*)[Ns]...) -> A<T, Ts...>' dependent trailing_return
// CHECK: |-InjectedClassNameType {{.*}} 'A<T, Ts...>' dependent
// CHECK: |-TemplateSpecializationType {{.*}} 'X<Ps...>' dependent X
// CHECK: | `-TemplateArgument expr
// CHECK: |   `-PackExpansionExpr {{.*}} 'T *'
// CHECK: |     `-DeclRefExpr {{.*}} 'T *' NonTypeTemplateParm {{.*}} 'Ps' 'T *'
// CHECK: `-PackExpansionType {{.*}} 'Ts (*)[Ns]...' dependent
// CHECK:   `-PointerType {{.*}} 'Ts (*)[Ns]' dependent contains_unexpanded_pack
// CHECK:     `-ParenType {{.*}} 'Ts [Ns]' sugar dependent contains_unexpanded_pack
// CHECK:       `-DependentSizedArrayType {{.*}} 'Ts [Ns]' dependent contains_unexpanded_pack
// CHECK:         |-TemplateTypeParmType {{.*}} 'Ts' dependent contains_unexpanded_pack depth 0 index 1 pack
// CHECK:         | `-TemplateTypeParm {{.*}} 'Ts'
// CHECK:         `-DeclRefExpr {{.*}} 'Ts' NonTypeTemplateParm {{.*}} 'Ns' 'Ts...'

template<typename T, T V> struct B {
  template<typename U, U W> B(X<W, V>);
};
B b(X<nullptr, 'x'>{});
using BT = decltype(b);
using BT = B<char, 'x'>;

// CHECK: Dumping <deduction guide for B>:
// CHECK: FunctionTemplateDecl
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'T' depth 0 index 1 V
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 2 U
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'type-parameter-0-2':'type-parameter-0-2' depth 0 index 3 W
// CHECK: |-CXXDeductionGuideDecl {{.*}} 'auto (X<W, V>) -> B<T, V>'
// CHECK: | `-ParmVarDecl {{.*}} 'X<W, V>'
// CHECK: `-CXXDeductionGuideDecl {{.*}} 'auto (X<nullptr, 'x'>) -> B<char, 'x'>'
// CHECK:   |-TemplateArgument type 'char'
// CHECK:   |-TemplateArgument integral 120
// CHECK:   |-TemplateArgument type 'nullptr_t'
// CHECK:   |-TemplateArgument nullptr
// CHECK:   `-ParmVarDecl {{.*}} 'X<nullptr, 'x'>':'X<nullptr, 'x'>'
// CHECK: FunctionProtoType {{.*}} 'auto (X<W, V>) -> B<T, V>' dependent trailing_return
// CHECK: |-InjectedClassNameType {{.*}} 'B<T, V>' dependent
// CHECK: `-TemplateSpecializationType {{.*}} 'X<W, V>' dependent X
// CHECK:   |-TemplateArgument expr
// CHECK:   | `-DeclRefExpr {{.*}} 'type-parameter-0-2':'type-parameter-0-2' NonTypeTemplateParm {{.*}} 'W' 'type-parameter-0-2':'type-parameter-0-2'
// CHECK:   `-TemplateArgument expr
// CHECK:     `-DeclRefExpr {{.*}} 'T' NonTypeTemplateParm {{.*}} 'V' 'T'
