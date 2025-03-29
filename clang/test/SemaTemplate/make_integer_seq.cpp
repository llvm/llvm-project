// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -ast-dump -verify -xc++ < %s | FileCheck %s

template <class A1, A1... A2> struct A {};

using test1 = __make_integer_seq<A, int, 1>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:5:1, col:43> col:7 test1 '__make_integer_seq<A, int, 1>':'A<int, 0>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, 1>' sugar
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, 1>' sugar alias
// CHECK-NEXT:       |-name: '__make_integer_seq' qualified
// CHECK-NEXT:       | `-BuiltinTemplateDecl {{.+}} __make_integer_seq
// CHECK-NEXT:       |-TemplateArgument template 'A'
// CHECK-NEXT:       | `-ClassTemplateDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:1, col:41> col:38 A
// CHECK-NEXT:       |-TemplateArgument type 'int'
// CHECK-NEXT:       | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:       |-TemplateArgument expr '1'
// CHECK-NEXT:       | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:42> 'int'
// CHECK-NEXT:       |   |-value: Int 1
// CHECK-NEXT:       |   `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:42> 'int' 1
// CHECK-NEXT:       `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} 'A<int, 0>' sugar
// CHECK-NEXT:         |-name: 'A' qualified
// CHECK-NEXT:         | `-ClassTemplateDecl {{.+}} A
// CHECK-NEXT:         |-TemplateArgument type 'int'
// CHECK-NEXT:         | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:         |-TemplateArgument expr '0'
// CHECK-NEXT:         | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:42> 'int'
// CHECK-NEXT:         |   |-value: Int 0
// CHECK-NEXT:         |   `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:42> 'int' 0
// CHECK-NEXT:         `-RecordType 0x{{[0-9A-Fa-f]+}} 'A<int, 0>'
// CHECK-NEXT:           `-ClassTemplateSpecialization 0x{{[0-9A-Fa-f]+}} 'A'

template <class B1, B1 B2> using B = __make_integer_seq<A, B1, B2>;
using test2 = B<int, 1>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:1, col:23> col:7 test2 'B<int, 1>':'A<int, 0>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} 'B<int, 1>' sugar
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} 'B<int, 1>' sugar alias
// CHECK-NEXT:       |-name: 'B' qualified
// CHECK-NEXT:       | `-TypeAliasTemplateDecl {{.+}} B
// CHECK-NEXT:       |-TemplateArgument type 'int'
// CHECK-NEXT:       | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:       |-TemplateArgument expr '1'
// CHECK-NEXT:       | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:22> 'int'
// CHECK-NEXT:       |   |-value: Int 1
// CHECK-NEXT:       |   `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:22> 'int' 1
// CHECK-NEXT:       `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, 1>' sugar
// CHECK-NEXT:         `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, 1>' sugar alias
// CHECK-NEXT:           |-name: '__make_integer_seq' qualified
// CHECK-NEXT:           | `-BuiltinTemplateDecl {{.+}} __make_integer_seq
// CHECK-NEXT:           |-TemplateArgument template 'A'
// CHECK-NEXT:           | `-ClassTemplateDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:1, col:41> col:38 A
// CHECK-NEXT:           |-TemplateArgument type 'int'
// CHECK-NEXT:           | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:           |-TemplateArgument expr '1'
// CHECK-NEXT:           | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:64> 'int'
// CHECK-NEXT:           |   |-value: Int 1
// CHECK-NEXT:           |   `-SubstNonTypeTemplateParmExpr 0x{{[0-9A-Fa-f]+}} <col:64> 'int'
// CHECK-NEXT:           |     |-NonTypeTemplateParmDecl 0x{{[0-9A-Fa-f]+}} <col:21, col:24> col:24 referenced 'B1' depth 0 index 1 B2
// CHECK-NEXT:           |     `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:64> 'int' 1
// CHECK-NEXT:           `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} 'A<int, 0>' sugar
// CHECK-NEXT:             |-name: 'A' qualified
// CHECK-NEXT:             | `-ClassTemplateDecl {{.+}} A
// CHECK-NEXT:             |-TemplateArgument type 'int'
// CHECK-NEXT:             | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:             |-TemplateArgument expr '0'
// CHECK-NEXT:             | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:64> 'int'
// CHECK-NEXT:             |   |-value: Int 0
// CHECK-NEXT:             |   `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:64> 'int' 0
// CHECK-NEXT:             `-RecordType 0x{{[0-9A-Fa-f]+}} 'A<int, 0>'
// CHECK-NEXT:               `-ClassTemplateSpecialization 0x{{[0-9A-Fa-f]+}} 'A'

template <template <class T, T...> class S, class T, int N> struct C {
  using test3 = __make_integer_seq<S, T, N>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:3, col:43> col:9 test3 '__make_integer_seq<S, T, N>':'__make_integer_seq<template-parameter-0-0, type-parameter-0-1, N>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<S, T, N>' sugar dependent
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<S, T, N>' sugar dependent alias
// CHECK-NEXT:       |-name: '__make_integer_seq' qualified
// CHECK-NEXT:       | `-BuiltinTemplateDecl {{.+}} __make_integer_seq
// CHECK-NEXT:       |-TemplateArgument template 'S'
// CHECK-NEXT:       | | `-TemplateTemplateParmDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:11, col:42> col:42 depth 0 index 0 S
// CHECK-NEXT:       |-TemplateArgument type 'T'
// CHECK-NEXT:       | `-TemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'T' dependent depth 0 index 1
// CHECK-NEXT:       |   `-TemplateTypeParm 0x{{[0-9A-Fa-f]+}} 'T'
// CHECK-NEXT:       |-TemplateArgument expr 'N'
// CHECK-NEXT:       | `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:42> 'T' <Dependent>
// CHECK-NEXT:       |   `-DeclRefExpr 0x{{[0-9A-Fa-f]+}} <col:42> 'int' NonTypeTemplateParm 0x{{[0-9A-Fa-f]+}} 'N' 'int'
// CHECK-NEXT:       `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<template-parameter-0-0, type-parameter-0-1, N>' dependent
// CHECK-NEXT:         |-name: '__make_integer_seq'
// CHECK-NEXT:         | `-BuiltinTemplateDecl {{.+}} __make_integer_seq
// CHECK-NEXT:         |-TemplateArgument template 'template-parameter-0-0'
// CHECK-NEXT:         | `-TemplateTemplateParmDecl 0x{{[0-9A-Fa-f]+}} <<invalid sloc>> <invalid sloc> depth 0 index 0
// CHECK-NEXT:         |-TemplateArgument type 'type-parameter-0-1'
// CHECK-NEXT:         | `-TemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'type-parameter-0-1' dependent depth 0 index 1
// CHECK-NEXT:         `-TemplateArgument expr 'N'
// CHECK-NEXT:           `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:42> 'T' <Dependent>
// CHECK-NEXT:             `-DeclRefExpr 0x{{[0-9A-Fa-f]+}} <col:42> 'int' NonTypeTemplateParm 0x{{[0-9A-Fa-f]+}} 'N' 'int'

  using test4 = __make_integer_seq<A, T, 1>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:3, col:43> col:9 test4 '__make_integer_seq<A, T, 1>':'__make_integer_seq<A, type-parameter-0-1, 1>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, T, 1>' sugar dependent
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, T, 1>' sugar dependent alias
// CHECK-NEXT:       |-name: '__make_integer_seq' qualified
// CHECK-NEXT:       | `-BuiltinTemplateDecl {{.+}} __make_integer_seq
// CHECK-NEXT:       |-TemplateArgument template 'A'
// CHECK-NEXT:       | `-ClassTemplateDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:1, col:41> col:38 A
// CHECK-NEXT:       |-TemplateArgument type 'T'
// CHECK-NEXT:       | `-TemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'T' dependent depth 0 index 1
// CHECK-NEXT:       |   `-TemplateTypeParm 0x{{[0-9A-Fa-f]+}} 'T'
// CHECK-NEXT:       |-TemplateArgument expr '1'
// CHECK-NEXT:       | `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:42> 'T' <Dependent>
// CHECK-NEXT:       |   `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:42> 'int' 1
// CHECK-NEXT:       `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, type-parameter-0-1, 1>' dependent
// CHECK-NEXT:         |-name: '__make_integer_seq'
// CHECK-NEXT:         | `-BuiltinTemplateDecl {{.+}} __make_integer_seq
// CHECK-NEXT:         |-TemplateArgument template 'A'
// CHECK-NEXT:         | `-ClassTemplateDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:1, col:41> col:38 A
// CHECK-NEXT:         |-TemplateArgument type 'type-parameter-0-1'
// CHECK-NEXT:         | `-TemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'type-parameter-0-1' dependent depth 0 index 1
// CHECK-NEXT:         `-TemplateArgument expr '1'
// CHECK-NEXT:           `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:42> 'T' <Dependent>
// CHECK-NEXT:             `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:42> 'int' 1

  using test5 = __make_integer_seq<A, int, N>;
//      CHECK: `-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:3, col:45> col:9 test5 '__make_integer_seq<A, int, N>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, N>' sugar dependent
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, N>' sugar dependent alias
// CHECK-NEXT:       |-name: '__make_integer_seq' qualified
// CHECK-NEXT:       | `-BuiltinTemplateDecl {{.+}} __make_integer_seq
// CHECK-NEXT:       |-TemplateArgument template 'A'
// CHECK-NEXT:       | `-ClassTemplateDecl 0x{{.+}} <line:{{.+}}:1, col:41> col:38 A
// CHECK-NEXT:       |-TemplateArgument type 'int'
// CHECK-NEXT:       | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:       |-TemplateArgument expr 'N'
// CHECK-NEXT:       | `-DeclRefExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:44> 'int' NonTypeTemplateParm 0x{{[0-9A-Fa-f]+}} 'N' 'int'
// CHECK-NEXT:       `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, N>' dependent
// CHECK-NEXT:         |-name: '__make_integer_seq'
// CHECK-NEXT:         | `-BuiltinTemplateDecl {{.+}} __make_integer_seq
// CHECK-NEXT:         |-TemplateArgument template 'A'
// CHECK-NEXT:         | `-ClassTemplateDecl 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:1, col:41> col:38 A
// CHECK-NEXT:         |-TemplateArgument type 'int'
// CHECK-NEXT:         | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:         `-TemplateArgument expr 'N'
// CHECK-NEXT:           `-DeclRefExpr 0x{{[0-9A-Fa-f]+}} <line:{{.+}}:44> 'int' NonTypeTemplateParm 0x{{[0-9A-Fa-f]+}} 'N' 'int'
};

// expected-no-diagnostics

template <class T, class S> struct D;
template <class T> struct D<T, __make_integer_seq<A, int, sizeof(T)>> {};
template struct D<char, A<int, 0>>;

template <class T, class S> struct E;
template <class T> struct E<T, __make_integer_seq<A, T, 2>> {};
template struct E<short, A<short, 0, 1>>;

template <template <class A1, A1... A2> class T, class S> struct F;
template <template <class A1, A1... A2> class T> struct F<T, __make_integer_seq<T, long, 3>> {};
template struct F<A, A<long, 0, 1, 2>>;

template <class T> struct G;
template <class T> struct G<__make_integer_seq<A, T, 1>> {};
template <class T> struct G<__make_integer_seq<A, T, 1U>> {};

template <int S, class = __make_integer_seq<A, int, S>> struct H;
template <int S, int... Is> struct H<S, A<int, Is...>> { };

template <int S> void h(H<S>);
void test_h() { h(H<5>{}); }
