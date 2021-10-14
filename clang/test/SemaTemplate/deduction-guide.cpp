// RUN: %clang_cc1 -std=c++2a -verify -ast-dump -ast-dump-decl-types -ast-dump-filter "deduction guide" %s | FileCheck %s --strict-whitespace

template<auto ...> struct X {};
template<template<typename X, X> typename> struct Y {};
template<typename ...> struct Z {};

template<typename T, typename ...Ts> struct A {
  template<Ts ...Ns, T *...Ps> A(X<Ps...>, Ts (*...qs)[Ns]);
};
int arr1[3], arr2[3];
short arr3[4];
A a(X<&arr1, &arr2>{}, &arr1, &arr2, &arr3);
using AT = decltype(a);
using AT = A<int[3], int, int, short>;

// CHECK-LABEL: Dumping <deduction guide for A>:
// CHECK: FunctionTemplateDecl
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 1 ... Ts
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'Ts...' depth 0 index 2 ... Ns
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'T *' depth 0 index 3 ... Ps
// CHECK: |-CXXDeductionGuideDecl
// CHECK: | |-ParmVarDecl {{.*}} 'X<Ps...>'
// CHECK: | `-ParmVarDecl {{.*}} 'Ts (*)[Ns]...' pack
// CHECK: `-CXXDeductionGuideDecl
// CHECK:   |-TemplateArgument type 'int[3]'
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
// CHECK:   | | `-Var {{.*}} 'arr1' 'int[3]'
// CHECK:   | `-TemplateArgument decl
// CHECK:   |   `-Var {{.*}} 'arr2' 'int[3]'
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
// CHECK:     `-ParenType {{.*}} 'Ts[Ns]' sugar dependent contains_unexpanded_pack
// CHECK:       `-DependentSizedArrayType {{.*}} 'Ts[Ns]' dependent contains_unexpanded_pack
// CHECK:         |-TemplateTypeParmType {{.*}} 'Ts' dependent contains_unexpanded_pack depth 0 index 1 pack
// CHECK:         | `-TemplateTypeParm {{.*}} 'Ts'
// CHECK:         `-DeclRefExpr {{.*}} 'Ts' NonTypeTemplateParm {{.*}} 'Ns' 'Ts...'

template<typename T, T V> struct B {
  template<typename U, U W> B(X<W, V>);
};
B b(X<nullptr, 'x'>{});
using BT = decltype(b);
using BT = B<char, 'x'>;

// CHECK-LABEL: Dumping <deduction guide for B>:
// CHECK: FunctionTemplateDecl
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'T' depth 0 index 1 V
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 2 U
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'type-parameter-0-2' depth 0 index 3 W
// CHECK: |-CXXDeductionGuideDecl {{.*}} 'auto (X<W, V>) -> B<T, V>'
// CHECK: | `-ParmVarDecl {{.*}} 'X<W, V>'
// CHECK: `-CXXDeductionGuideDecl {{.*}} 'auto (X<nullptr, 'x'>) -> B<char, 'x'>'
// CHECK:   |-TemplateArgument type 'char'
// CHECK:   |-TemplateArgument integral 120
// CHECK:   |-TemplateArgument type 'std::nullptr_t'
// CHECK:   |-TemplateArgument nullptr
// CHECK:   `-ParmVarDecl {{.*}} 'X<nullptr, 'x'>':'X<nullptr, 'x'>'
// CHECK: FunctionProtoType {{.*}} 'auto (X<W, V>) -> B<T, V>' dependent trailing_return
// CHECK: |-InjectedClassNameType {{.*}} 'B<T, V>' dependent
// CHECK: `-TemplateSpecializationType {{.*}} 'X<W, V>' dependent X
// CHECK:   |-TemplateArgument expr
// CHECK:   | `-DeclRefExpr {{.*}} 'type-parameter-0-2' NonTypeTemplateParm {{.*}} 'W' 'type-parameter-0-2'
// CHECK:   `-TemplateArgument expr
// CHECK:     `-DeclRefExpr {{.*}} 'T' NonTypeTemplateParm {{.*}} 'V' 'T'

template<typename A> struct C {
  template<template<typename X, X> typename T, typename U, U V = 0> C(A, Y<T>, U);
};
C c(1, Y<B>{}, 2);
using CT = decltype(c);
using CT = C<int>;

// CHECK-LABEL: Dumping <deduction guide for C>:
// CHECK: FunctionTemplateDecl
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 A
// CHECK: |-TemplateTemplateParmDecl {{.*}} depth 0 index 1 T
// CHECK: | |-TemplateTypeParmDecl {{.*}} typename depth 1 index 0 X
// CHECK: | `-NonTypeTemplateParmDecl {{.*}} 'X' depth 1 index 1
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 2 U
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'type-parameter-0-2' depth 0 index 3 V
// CHECK: | `-TemplateArgument expr
// CHECK: |   `-IntegerLiteral {{.*}} 'int' 0
// CHECK: |-CXXDeductionGuideDecl {{.*}} 'auto (A, Y<>, type-parameter-0-2) -> C<A>'
// CHECK: | |-ParmVarDecl {{.*}} 'A'
// CHECK: | |-ParmVarDecl {{.*}} 'Y<>'
// CHECK: | `-ParmVarDecl {{.*}} 'type-parameter-0-2'
// CHECK: `-CXXDeductionGuideDecl {{.*}} 'auto (int, Y<B>, int) -> C<int>'
// CHECK:  |-TemplateArgument type 'int'
// CHECK:  |-TemplateArgument template B
// CHECK:  |-TemplateArgument type 'int'
// CHECK:  |-TemplateArgument integral 0
// CHECK:  |-ParmVarDecl {{.*}} 'int':'int'
// CHECK:  |-ParmVarDecl {{.*}} 'Y<B>':'Y<B>'
// CHECK:  `-ParmVarDecl {{.*}} 'int':'int'
// CHECK: FunctionProtoType {{.*}} 'auto (A, Y<>, type-parameter-0-2) -> C<A>' dependent trailing_return cdecl
// CHECK: |-InjectedClassNameType {{.*}} 'C<A>' dependent
// CHECK: |-TemplateTypeParmType {{.*}} 'A' dependent depth 0 index 0
// CHECK: | `-TemplateTypeParm {{.*}} 'A'
// CHECK: |-TemplateSpecializationType {{.*}} 'Y<>' dependent Y
// CHECK: | `-TemplateArgument template 
// CHECK: `-TemplateTypeParmType {{.*}} 'type-parameter-0-2' dependent depth 0 index 2

template<typename ...T> struct D { // expected-note {{candidate}}
  template<typename... U> using B = int(int (*...p)(T, U));
  template<typename U1, typename U2> D(B<U1, U2>*); // expected-note {{candidate}}
};
int f(int(int, int), int(int, int));
// FIXME: We can't deduce this because we can't deduce through a
// SubstTemplateTypeParmPackType.
D d = f; // expected-error {{no viable}}
using DT = decltype(d);
using DT = D<int, int>;

// CHECK-LABEL: Dumping <deduction guide for D>:
// CHECK: FunctionTemplateDecl
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 ... T
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 1 U1
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 2 U2
// CHECK: `-CXXDeductionGuideDecl {{.*}} 'auto (B<type-parameter-0-1, type-parameter-0-2> *) -> D<T...>'  
// CHECK:   `-ParmVarDecl {{.*}} 'B<type-parameter-0-1, type-parameter-0-2> *'
// CHECK: FunctionProtoType {{.*}} 'auto (B<type-parameter-0-1, type-parameter-0-2> *) -> D<T...>' dependent trailing_return
// CHECK: |-InjectedClassNameType {{.*}} 'D<T...>' dependent
// CHECK: `-PointerType {{.*}} 'B<type-parameter-0-1, type-parameter-0-2> *' dependent
// CHECK:   `-TemplateSpecializationType {{.*}} 'B<type-parameter-0-1, type-parameter-0-2>' sugar dependent alias B
// CHECK:     |-TemplateArgument type 'type-parameter-0-1'
// CHECK:     |-TemplateArgument type 'type-parameter-0-2'
// CHECK:     `-FunctionProtoType {{.*}} 'int (int (*)(T, U)...)' dependent cdecl
// CHECK:       |-BuiltinType {{.*}} 'int'
// CHECK:       `-PackExpansionType {{.*}} 'int (*)(T, U)...' dependent expansions 2
// CHECK:         `-PointerType {{.*}} 'int (*)(T, U)' dependent contains_unexpanded_pack
// CHECK:           `-ParenType {{.*}} 'int (T, U)' sugar dependent contains_unexpanded_pack
// CHECK:             `-FunctionProtoType {{.*}} 'int (T, U)' dependent contains_unexpanded_pack cdecl
// CHECK:               |-BuiltinType {{.*}} 'int'
// CHECK:               |-TemplateTypeParmType {{.*}} 'T' dependent contains_unexpanded_pack depth 0 index 0 pack
// CHECK:               | `-TemplateTypeParm {{.*}} 'T'
// CHECK:               `-SubstTemplateTypeParmPackType {{.*}} 'U' dependent contains_unexpanded_pack
// CHECK:                 |-TemplateTypeParmType {{.*}} 'U' dependent contains_unexpanded_pack depth 1 index 0 pack
// CHECK:                 | `-TemplateTypeParm {{.*}} 'U'
// CHECK:                 `-TemplateArgument pack
// CHECK:                   |-TemplateArgument type 'type-parameter-0-1'
// CHECK-NOT: Subst
// CHECK:                   | `-TemplateTypeParmType
// CHECK:                   `-TemplateArgument type 'type-parameter-0-2'
// CHECK-NOT: Subst
// CHECK:                     `-TemplateTypeParmType

template<int ...N> struct E { // expected-note {{candidate}}
  template<int ...M> using B = Z<X<N, M>...>;
  template<int M1, int M2> E(B<M1, M2>); // expected-note {{candidate}}
};
// FIXME: We can't deduce this because we can't deduce through a
// SubstNonTypeTemplateParmPackExpr.
E e = Z<X<1, 2>, X<3, 4>>(); // expected-error {{no viable}}
using ET = decltype(e);
using ET = E<1, 3>;

// CHECK-LABEL: Dumping <deduction guide for E>:
// CHECK: FunctionTemplateDecl
// CHECK: |-NonTypeTemplateParmDecl [[N:0x[0-9a-f]*]] {{.*}} 'int' depth 0 index 0 ... N
// CHECK: |-NonTypeTemplateParmDecl [[M1:0x[0-9a-f]*]] {{.*}} 'int' depth 0 index 1 M1
// CHECK: |-NonTypeTemplateParmDecl [[M2:0x[0-9a-f]*]] {{.*}} 'int' depth 0 index 2 M2
// CHECK: `-CXXDeductionGuideDecl {{.*}} 'auto (B<M1, M2>) -> E<N...>'
// CHECK:   `-ParmVarDecl {{.*}} 'B<M1, M2>':'Z<X<N, M>...>'
// CHECK: FunctionProtoType {{.*}} 'auto (B<M1, M2>) -> E<N...>' dependent trailing_return
// CHECK: |-InjectedClassNameType {{.*}} 'E<N...>' dependent
// CHECK: `-TemplateSpecializationType {{.*}} 'B<M1, M2>' sugar dependent alias B
// CHECK:   |-TemplateArgument expr
// CHECK:   | `-DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'M1' 'int'
// CHECK:   |-TemplateArgument expr
// CHECK:   | `-DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'M2' 'int'
// CHECK:   `-TemplateSpecializationType {{.*}} 'Z<X<N, M>...>' dependent Z
// CHECK:     `-TemplateArgument type 'X<N, M>...'
// CHECK:       `-PackExpansionType {{.*}} 'X<N, M>...' dependent expansions 2
// CHECK:         `-TemplateSpecializationType {{.*}} 'X<N, M>' dependent contains_unexpanded_pack X
// CHECK:           |-TemplateArgument expr
// CHECK-NOT: Subst
// CHECK:           | `-DeclRefExpr {{.*}} 'int' NonTypeTemplateParm [[N]] 'N' 'int'
// CHECK:           `-TemplateArgument expr
// CHECK:             `-SubstNonTypeTemplateParmPackExpr {{.*}} 'int'
// CHECK:               |-NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 1 index 0 ... M
// CHECK:               `-TemplateArgument pack
// CHECK:                 |-TemplateArgument expr
// CHECK-NOT: Subst
// CHECK:                 | `-DeclRefExpr {{.*}} 'int' NonTypeTemplateParm [[M1]] 'M1' 'int'
// CHECK:                 `-TemplateArgument expr
// CHECK-NOT: Subst
// CHECK:                   `-DeclRefExpr {{.*}} 'int' NonTypeTemplateParm [[M2]] 'M2' 'int'
