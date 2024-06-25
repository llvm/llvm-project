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
// CHECK:   | |-TemplateArgument integral '3'
// CHECK:   | |-TemplateArgument integral '3'
// CHECK:   | `-TemplateArgument integral '(short)4'
// CHECK:   |-TemplateArgument pack
// CHECK:   | |-TemplateArgument decl
// CHECK:   | | `-Var {{.*}} 'arr1' 'int[3]'
// CHECK:   | `-TemplateArgument decl
// CHECK:   |   `-Var {{.*}} 'arr2' 'int[3]'
// CHECK:   |-ParmVarDecl {{.*}} 'X<&arr1, &arr2>'
// CHECK:   |-ParmVarDecl {{.*}} 'int (*)[3]'
// CHECK:   |-ParmVarDecl {{.*}} 'int (*)[3]'
// CHECK:   `-ParmVarDecl {{.*}} 'short (*)[4]'
// CHECK: FunctionProtoType {{.*}} 'auto (X<Ps...>, Ts (*)[Ns]...) -> A<T, Ts...>' dependent trailing_return
// CHECK: |-InjectedClassNameType {{.*}} 'A<T, Ts...>' dependent
// CHECK: |-ElaboratedType {{.*}} 'X<Ps...>' sugar dependent
// CHECK: | `-TemplateSpecializationType {{.*}} 'X<Ps...>' dependent
// CHECK: |   `-TemplateArgument expr
// CHECK: |     `-PackExpansionExpr {{.*}} 'T *'
// CHECK: |       `-DeclRefExpr {{.*}} 'T *' NonTypeTemplateParm {{.*}} 'Ps' 'T *'
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
// CHECK:   |-TemplateArgument integral ''x''
// CHECK:   |-TemplateArgument type 'std::nullptr_t'
// CHECK:   |-TemplateArgument nullptr
// CHECK:   `-ParmVarDecl {{.*}} 'X<nullptr, 'x'>'
// CHECK: FunctionProtoType {{.*}} 'auto (X<W, V>) -> B<T, V>' dependent trailing_return
// CHECK: |-InjectedClassNameType {{.*}} 'B<T, V>' dependent
// CHECK: `-TemplateSpecializationType {{.*}} 'X<W, V>' dependent
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
// CHECK: | `-TemplateArgument {{.*}} expr
// CHECK: |   `-IntegerLiteral {{.*}} 'int' 0
// CHECK: |-CXXDeductionGuideDecl {{.*}} 'auto (A, Y<template-parameter-0-1>, type-parameter-0-2) -> C<A>'
// CHECK: | |-ParmVarDecl {{.*}} 'A'
// CHECK: | |-ParmVarDecl {{.*}} 'Y<template-parameter-0-1>'
// CHECK: | `-ParmVarDecl {{.*}} 'type-parameter-0-2'
// CHECK: `-CXXDeductionGuideDecl {{.*}} 'auto (int, Y<B>, int) -> C<int>'
// CHECK:  |-TemplateArgument type 'int'
// CHECK:  |-TemplateArgument template 'B'
// CHECK:  |-TemplateArgument type 'int'
// CHECK:  |-TemplateArgument integral '0'
// CHECK:  |-ParmVarDecl {{.*}} 'int'
// CHECK:  |-ParmVarDecl {{.*}} 'Y<B>'
// CHECK:  `-ParmVarDecl {{.*}} 'int'
// CHECK: FunctionProtoType {{.*}} 'auto (A, Y<template-parameter-0-1>, type-parameter-0-2) -> C<A>' dependent trailing_return cdecl
// CHECK: |-InjectedClassNameType {{.*}} 'C<A>' dependent
// CHECK: |-TemplateTypeParmType {{.*}} 'A' dependent depth 0 index 0
// CHECK: | `-TemplateTypeParm {{.*}} 'A'
// CHECK: |-ElaboratedType {{.*}} 'Y<template-parameter-0-1>' sugar dependent
// CHECK: | `-TemplateSpecializationType {{.*}} 'Y<template-parameter-0-1>' dependent
// CHECK: |   `-TemplateArgument template
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
// CHECK:   `-TemplateSpecializationType {{.*}} 'B<type-parameter-0-1, type-parameter-0-2>' sugar dependent alias
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
// CHECK:               `-SubstTemplateTypeParmPackType {{.*}} 'U' dependent contains_unexpanded_pack typename depth 1 index 0 ... U
// CHECK:                 |-TypeAliasTemplate {{.*}} 'B'
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
// CHECK: `-TemplateSpecializationType {{.*}} 'B<M1, M2>' sugar dependent alias
// CHECK:   |-TemplateArgument expr
// CHECK:   | `-DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'M1' 'int'
// CHECK:   |-TemplateArgument expr
// CHECK:   | `-DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'M2' 'int'
// CHECK:   `-TemplateSpecializationType {{.*}} 'Z<X<N, M>...>' dependent
// CHECK:     `-TemplateArgument type 'X<N, M>...'
// CHECK:       `-PackExpansionType {{.*}} 'X<N, M>...' dependent expansions 2
// CHECK:         `-TemplateSpecializationType {{.*}} 'X<N, M>' dependent contains_unexpanded_pack
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

template <char = 'x'> struct F;

template <char> struct F {
  template <typename U>
  requires(false) F(U);
  template <typename U>
  requires(true) F(U);
};

F s(0);

// CHECK-LABEL: Dumping <deduction guide for F>:
// CHECK: FunctionTemplateDecl
// CHECK: |-NonTypeTemplateParmDecl {{.*}} 'char' depth 0 index 0
// CHECK:   `-TemplateArgument {{.*}} expr
// CHECK: |   |-inherited from NonTypeTemplateParm {{.*}} '' 'char'
// CHECK: |   `-CharacterLiteral {{.*}} 'char' 120
// CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 1 U
// CHECK: |-ParenExpr {{.*}} 'bool'
// CHECK: | `-CXXBoolLiteralExpr {{.*}} 'bool' false
// CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for F> 'auto (type-parameter-0-1) -> F<>'
// CHECK: | `-ParmVarDecl {{.*}} 'type-parameter-0-1'
// CHECK: `-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for F> 'auto (int) -> F<>'
// CHECK:   |-TemplateArgument integral ''x''
// CHECK:   |-TemplateArgument type 'int'
// CHECK:   | `-BuiltinType {{.*}} 'int'
// CHECK:   `-ParmVarDecl {{.*}} 'int'
// CHECK: FunctionProtoType {{.*}} 'auto (type-parameter-0-1) -> F<>' dependent trailing_return cdecl
// CHECK: |-InjectedClassNameType {{.*}} 'F<>' dependent
// CHECK: | `-CXXRecord {{.*}} 'F'
// CHECK: `-TemplateTypeParmType {{.*}} 'type-parameter-0-1' dependent depth 0 index 1

template<typename T>
struct G { T t; };

G g = {1};
// CHECK-LABEL: Dumping <deduction guide for G>:
// CHECK: FunctionTemplateDecl
// CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for G> 'auto (T) -> G<T>' aggregate
// CHECK: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for G> 'auto (int) -> G<int>' implicit_instantiation aggregate

template<typename X>
using AG = G<X>;
AG ag = {1};
// Verify that the aggregate deduction guide for alias templates is built.
// CHECK-LABEL: Dumping <deduction guide for AG>
// CHECK: FunctionTemplateDecl
// CHECK: |-CXXDeductionGuideDecl {{.*}} 'auto (type-parameter-0-0) -> G<type-parameter-0-0>'
// CHECK: `-CXXDeductionGuideDecl {{.*}} 'auto (int) -> G<int>' implicit_instantiation
// CHECK:   |-TemplateArgument type 'int'
// CHECK:   | `-BuiltinType {{.*}} 'int'
// CHECK:   `-ParmVarDecl {{.*}} 'int'

template <typename X = int>
using BG = G<int>;
BG bg(1.0);
// CHECK-LABEL: Dumping <deduction guide for BG>
// CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for BG>
// CHECK: |-CXXDeductionGuideDecl {{.*}} 'auto (int) -> G<int>' aggregate

template <typename D>
requires (sizeof(D) == 4)
struct Foo {
  Foo(D);
};

template <typename U>
using AFoo = Foo<G<U>>;
// Verify that the require-clause from the Foo deduction guide is transformed.
// The D occurrence should be rewritten to G<type-parameter-0-0>.
//
// CHECK-LABEL: Dumping <deduction guide for AFoo>
// CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for AFoo>
// CHECK-NEXT: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 U
// CHECK-NEXT: |-BinaryOperator {{.*}} '&&'
// CHECK-NEXT: | |-ParenExpr {{.*}} 'bool'
// CHECK-NEXT: | | `-BinaryOperator {{.*}} 'bool' '=='
// CHECK-NEXT: | |   |-UnaryExprOrTypeTraitExpr {{.*}} 'G<type-parameter-0-0>'
// CHECK-NEXT: | |   `-ImplicitCastExpr {{.*}}
// CHECK-NEXT: | |     `-IntegerLiteral {{.*}}
// CHECK-NEXT: | `-TypeTraitExpr {{.*}} 'bool' __is_deducible
// CHECK-NEXT: |   |-DeducedTemplateSpecializationType {{.*}} 'AFoo' dependent
// CHECK-NEXT: |   | `-name: 'AFoo'
// CHECK-NEXT: |   |   `-TypeAliasTemplateDecl {{.+}} AFoo
// CHECK-NEXT: |   `-TemplateSpecializationType {{.*}} 'Foo<G<type-parameter-0-0>>' dependent
// CHECK:      |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for AFoo> 'auto (G<type-parameter-0-0>) -> Foo<G<type-parameter-0-0>>'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} 'G<type-parameter-0-0>'
// CHECK-NEXT: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for AFoo> 'auto (G<int>) -> Foo<G<int>>' implicit_instantiation
// CHECK-NEXT:   |-TemplateArgument type 'int'
// CHECK-NEXT:   | `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:   `-ParmVarDecl {{.*}} 'G<int>'

AFoo aa(G<int>{});

namespace TTP {
  template<typename> struct A {};

  template<class T> struct B {
    template<template <class> typename TT> B(TT<T>);
  };

  B b(A<int>{});
} // namespace TTP

// CHECK-LABEL: Dumping TTP::<deduction guide for B>:
// CHECK-NEXT:  FunctionTemplateDecl 0x{{.+}} <{{.+}}:[[# @LINE - 7]]:5, col:51>
// CHECK-NEXT:  |-TemplateTypeParmDecl {{.+}} class depth 0 index 0 T{{$}}
// CHECK-NEXT:  |-TemplateTemplateParmDecl {{.+}} depth 0 index 1 TT{{$}}
// CHECK-NEXT:  | `-TemplateTypeParmDecl {{.+}} class depth 1 index 0{{$}}
// CHECK-NEXT:  |-CXXDeductionGuideDecl {{.+}} 'auto (template-parameter-0-1<T>) -> B<T>'{{$}}
// CHECK-NEXT:  | `-ParmVarDecl {{.+}} 'template-parameter-0-1<T>'{{$}}
// CHECK-NEXT:  `-CXXDeductionGuideDecl {{.+}} 'auto (A<int>) -> TTP::B<int>'
// CHECK-NEXT:    |-TemplateArgument type 'int'
// CHECK-NEXT:    | `-BuiltinType {{.+}} 'int'{{$}}
// CHECK-NEXT:    |-TemplateArgument template 'TTP::A'{{$}}
// CHECK-NEXT:    | `-ClassTemplateDecl {{.+}} A{{$}}
// CHECK-NEXT:    `-ParmVarDecl {{.+}} 'A<int>':'TTP::A<int>'{{$}}
// CHECK-NEXT:  FunctionProtoType {{.+}} 'auto (template-parameter-0-1<T>) -> B<T>' dependent trailing_return cdecl{{$}}
// CHECK-NEXT:  |-InjectedClassNameType {{.+}} 'B<T>' dependent{{$}}
// CHECK-NEXT:  | `-CXXRecord {{.+}} 'B'{{$}}
// CHECK-NEXT:  `-ElaboratedType {{.+}} 'template-parameter-0-1<T>' sugar dependent{{$}}
// CHECK-NEXT:    `-TemplateSpecializationType {{.+}} 'template-parameter-0-1<T>' dependent{{$}}
// CHECK-NEXT:      |-name: 'template-parameter-0-1' qualified
// CHECK-NEXT:      | `-TemplateTemplateParmDecl {{.+}} depth 0 index 1
// CHECK-NEXT:      `-TemplateArgument type 'T':'type-parameter-0-0'{{$}}
// CHECK-NEXT:        `-TemplateTypeParmType {{.+}} 'T' dependent depth 0 index 0{{$}}
// CHECK-NEXT:          `-TemplateTypeParm {{.+}} 'T'{{$}}

namespace GH64625 {

template <class T> struct X {
  T t[2];
};

X x = {{1, 2}};

// CHECK-LABEL: Dumping GH64625::<deduction guide for X>:
// CHECK-NEXT: FunctionTemplateDecl {{.+}} <{{.+}}:[[#@LINE - 7]]:1, col:27> col:27 implicit <deduction guide for X>
// CHECK-NEXT: |-TemplateTypeParmDecl {{.+}} <col:11, col:17> col:17 referenced class depth 0 index 0 T
// CHECK:      |-CXXDeductionGuideDecl {{.+}} <col:27> col:27 implicit <deduction guide for X> 'auto (T (&&)[2]) -> X<T>' aggregate 
// CHECK-NEXT: | `-ParmVarDecl {{.+}} <col:27> col:27 'T (&&)[2]'
// CHECK-NEXT: `-CXXDeductionGuideDecl {{.+}} <col:27> col:27 implicit used <deduction guide for X> 'auto (int (&&)[2]) -> GH64625::X<int>' implicit_instantiation aggregate 
// CHECK-NEXT:  |-TemplateArgument type 'int'
// CHECK-NEXT:  | `-BuiltinType {{.+}} 'int'
// CHECK-NEXT:  `-ParmVarDecl {{.+}} <col:27> col:27 'int (&&)[2]'
// CHECK-NEXT: FunctionProtoType {{.+}} 'auto (T (&&)[2]) -> X<T>' dependent trailing_return
// CHECK-NEXT: |-InjectedClassNameType {{.+}} 'X<T>' dependent
// CHECK-NEXT: | `-CXXRecord {{.+}} 'X'
// CHECK-NEXT: `-RValueReferenceType {{.+}} 'T (&&)[2]' dependent
// CHECK-NEXT:  `-ConstantArrayType {{.+}} 'T[2]' dependent 2 
// CHECK-NEXT:    `-TemplateTypeParmType {{.+}} 'T' dependent depth 0 index 0
// CHECK-NEXT:      `-TemplateTypeParm {{.+}} 'T'

template <class T, class U> struct TwoArrays {
  T t[2];
  U u[3];
};

TwoArrays ta = {{1, 2}, {3, 4, 5}};
// CHECK-LABEL: Dumping GH64625::<deduction guide for TwoArrays>:
// CHECK-NEXT: FunctionTemplateDecl {{.+}} <{{.+}}:[[#@LINE - 7]]:1, col:36> col:36 implicit <deduction guide for TwoArrays> 
// CHECK-NEXT: |-TemplateTypeParmDecl {{.+}} <col:11, col:17> col:17 referenced class depth 0 index 0 T 
// CHECK-NEXT: |-TemplateTypeParmDecl {{.+}} <col:20, col:26> col:26 referenced class depth 0 index 1 U 
// CHECK:      |-CXXDeductionGuideDecl {{.+}} <col:36> col:36 implicit <deduction guide for TwoArrays> 'auto (T (&&)[2], U (&&)[3]) -> TwoArrays<T, U>' aggregate  
// CHECK-NEXT: | |-ParmVarDecl {{.+}} <col:36> col:36 'T (&&)[2]' 
// CHECK-NEXT: | `-ParmVarDecl {{.+}} <col:36> col:36 'U (&&)[3]' 
// CHECK-NEXT: `-CXXDeductionGuideDecl {{.+}} <col:36> col:36 implicit used <deduction guide for TwoArrays> 'auto (int (&&)[2], int (&&)[3]) -> GH64625::TwoArrays<int, int>' implicit_instantiation aggregate  
// CHECK-NEXT:   |-TemplateArgument type 'int' 
// CHECK-NEXT:   | `-BuiltinType {{.+}} 'int' 
// CHECK-NEXT:   |-TemplateArgument type 'int' 
// CHECK-NEXT:   | `-BuiltinType {{.+}} 'int' 
// CHECK-NEXT:   |-ParmVarDecl {{.+}} <col:36> col:36 'int (&&)[2]' 
// CHECK-NEXT:   `-ParmVarDecl {{.+}} <col:36> col:36 'int (&&)[3]' 
// CHECK-NEXT: FunctionProtoType {{.+}} 'auto (T (&&)[2], U (&&)[3]) -> TwoArrays<T, U>' dependent trailing_return 
// CHECK-NEXT: |-InjectedClassNameType {{.+}} 'TwoArrays<T, U>' dependent 
// CHECK-NEXT: | `-CXXRecord {{.+}} 'TwoArrays' 
// CHECK-NEXT: |-RValueReferenceType {{.+}} 'T (&&)[2]' dependent 
// CHECK-NEXT: | `-ConstantArrayType {{.+}} 'T[2]' dependent 2  
// CHECK-NEXT: |   `-TemplateTypeParmType {{.+}} 'T' dependent depth 0 index 0 
// CHECK-NEXT: |     `-TemplateTypeParm {{.+}} 'T' 
// CHECK-NEXT: `-RValueReferenceType {{.+}} 'U (&&)[3]' dependent 
// CHECK-NEXT:   `-ConstantArrayType {{.+}} 'U[3]' dependent 3  
// CHECK-NEXT:     `-TemplateTypeParmType {{.+}} 'U' dependent depth 0 index 1 
// CHECK-NEXT:       `-TemplateTypeParm {{.+}} 'U'

TwoArrays tb = {1, 2, {3, 4, 5}};
// CHECK:   |-CXXDeductionGuideDecl {{.+}} <col:36> col:36 implicit <deduction guide for TwoArrays> 'auto (T, T, U (&&)[3]) -> TwoArrays<T, U>' aggregate 
// CHECK-NEXT: | |-ParmVarDecl {{.+}} <col:36> col:36 'T'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} <col:36> col:36 'T'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} <col:36> col:36 'U (&&)[3]'
// CHECK-NEXT: `-CXXDeductionGuideDecl {{.+}} <col:36> col:36 implicit used <deduction guide for TwoArrays> 'auto (int, int, int (&&)[3]) -> GH64625::TwoArrays<int, int>' implicit_instantiation aggregate 
// CHECK-NEXT:   |-TemplateArgument type 'int'
// CHECK-NEXT:   | `-BuiltinType {{.+}} 'int'
// CHECK-NEXT:   |-TemplateArgument type 'int'
// CHECK-NEXT:   | `-BuiltinType {{.+}} 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} <col:36> col:36 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} <col:36> col:36 'int'
// CHECK-NEXT:   `-ParmVarDecl {{.+}} <col:36> col:36 'int (&&)[3]'
// CHECK-NEXT: FunctionProtoType {{.+}} 'auto (T, T, U (&&)[3]) -> TwoArrays<T, U>' dependent trailing_return
// CHECK-NEXT: |-InjectedClassNameType {{.+}} 'TwoArrays<T, U>' dependent
// CHECK-NEXT: | `-CXXRecord {{.+}} 'TwoArrays'
// CHECK-NEXT: |-TemplateTypeParmType {{.+}} 'T' dependent depth 0 index 0
// CHECK-NEXT: | `-TemplateTypeParm {{.+}} 'T'
// CHECK-NEXT: |-TemplateTypeParmType {{.+}} 'T' dependent depth 0 index 0
// CHECK-NEXT: | `-TemplateTypeParm {{.+}} 'T'
// CHECK-NEXT: `-RValueReferenceType {{.+}} 'U (&&)[3]' dependent
// CHECK-NEXT:   `-ConstantArrayType {{.+}} 'U[3]' dependent 3 
// CHECK-NEXT:     `-TemplateTypeParmType {{.+}} 'U' dependent depth 0 index 1
// CHECK-NEXT:       `-TemplateTypeParm {{.+}} 'U'

TwoArrays tc = {{1, 2}, 3, 4, 5};
// CHECK: |-CXXDeductionGuideDecl {{.+}} <col:36> col:36 implicit <deduction guide for TwoArrays> 'auto (T (&&)[2], U, U, U) -> TwoArrays<T, U>' aggregate 
// CHECK-NEXT: | |-ParmVarDecl {{.+}} <col:36> col:36 'T (&&)[2]'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} <col:36> col:36 'U'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} <col:36> col:36 'U'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} <col:36> col:36 'U'
// CHECK-NEXT: `-CXXDeductionGuideDecl {{.+}} <col:36> col:36 implicit used <deduction guide for TwoArrays> 'auto (int (&&)[2], int, int, int) -> GH64625::TwoArrays<int, int>' implicit_instantiation aggregate 
// CHECK-NEXT:   |-TemplateArgument type 'int'
// CHECK-NEXT:   | `-BuiltinType {{.+}} 'int'
// CHECK-NEXT:   |-TemplateArgument type 'int'
// CHECK-NEXT:   | `-BuiltinType {{.+}} 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} <col:36> col:36 'int (&&)[2]'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} <col:36> col:36 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} <col:36> col:36 'int'
// CHECK-NEXT:   `-ParmVarDecl {{.+}} <col:36> col:36 'int'
// CHECK-NEXT: FunctionProtoType {{.+}} 'auto (T (&&)[2], U, U, U) -> TwoArrays<T, U>' dependent trailing_return
// CHECK-NEXT: |-InjectedClassNameType {{.+}} 'TwoArrays<T, U>' dependent
// CHECK-NEXT: | `-CXXRecord {{.+}} 'TwoArrays'
// CHECK-NEXT: |-RValueReferenceType {{.+}} 'T (&&)[2]' dependent
// CHECK-NEXT: | `-ConstantArrayType {{.+}} 'T[2]' dependent 2 
// CHECK-NEXT: |   `-TemplateTypeParmType {{.+}} 'T' dependent depth 0 index 0
// CHECK-NEXT: |     `-TemplateTypeParm {{.+}} 'T'
// CHECK-NEXT: |-TemplateTypeParmType {{.+}} 'U' dependent depth 0 index 1
// CHECK-NEXT: | `-TemplateTypeParm {{.+}} 'U'
// CHECK-NEXT: |-TemplateTypeParmType {{.+}} 'U' dependent depth 0 index 1
// CHECK-NEXT: | `-TemplateTypeParm {{.+}} 'U'
// CHECK-NEXT: `-TemplateTypeParmType {{.+}} 'U' dependent depth 0 index 1
// CHECK-NEXT:   `-TemplateTypeParm {{.+}} 'U'

} // namespace GH64625

namespace GH83368 {

template <int N> struct A {
  int f1[N];
};

A a{.f1 = {1}};

// CHECK-LABEL: Dumping GH83368::<deduction guide for A>:
// CHECK-NEXT: FunctionTemplateDecl 0x{{.+}} <{{.+}}:[[#@LINE - 7]]:1, col:25> col:25 implicit <deduction guide for A>
// CHECK-NEXT: |-NonTypeTemplateParmDecl {{.+}} <col:11, col:15> col:15 referenced 'int' depth 0 index 0 N
// CHECK:      |-CXXDeductionGuideDecl {{.+}} <col:25> col:25 implicit <deduction guide for A> 'auto (int (&&)[N]) -> A<N>' aggregate
// CHECK-NEXT: | `-ParmVarDecl {{.+}} <col:25> col:25 'int (&&)[N]'
// CHECK-NEXT: `-CXXDeductionGuideDecl {{.+}} <col:25> col:25 implicit used <deduction guide for A> 'auto (int (&&)[1]) -> GH83368::A<1>' implicit_instantiation aggregate 
// CHECK-NEXT:   |-TemplateArgument integral '1'
// CHECK-NEXT:   `-ParmVarDecl {{.+}} <col:25> col:25 'int (&&)[1]'
// CHECK-NEXT: FunctionProtoType {{.+}} 'auto (int (&&)[N]) -> A<N>' dependent trailing_return
// CHECK-NEXT: |-InjectedClassNameType {{.+}} 'A<N>' dependent
// CHECK-NEXT: | `-CXXRecord {{.+}} 'A'
// CHECK-NEXT: `-RValueReferenceType {{.+}} 'int (&&)[N]' dependent
// CHECK-NEXT:   `-DependentSizedArrayType {{.+}} 'int[N]' dependent
// CHECK-NEXT:     |-BuiltinType {{.+}} 'int'
// CHECK-NEXT:     `-DeclRefExpr {{.+}} <col:10> 'int' NonTypeTemplateParm {{.+}} 'N' 'int'

} // namespace GH83368
