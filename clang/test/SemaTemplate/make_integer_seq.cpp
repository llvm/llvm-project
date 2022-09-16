// RUN: %clang_cc1 -std=c++2b -fsyntax-only -triple x86_64-linux-gnu -ast-dump -verify -xc++ < %s | FileCheck %s

template <class A1, A1... A2> struct A {};

using test1 = __make_integer_seq<A, int, 1>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:5:1, col:43> col:7 test1 '__make_integer_seq<A, int, 1>':'A<int, 0>'
// CHECK-NEXT: | `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, 1>' sugar
// CHECK-NEXT: |   `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, 1>' sugar __make_integer_seq
// CHECK-NEXT: |     |-TemplateArgument template A
// CHECK-NEXT: |     |-TemplateArgument type 'int'
// CHECK-NEXT: |     | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT: |     |-TemplateArgument expr
// CHECK-NEXT: |     | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <col:42> 'int'
// CHECK-NEXT: |     |   |-value: Int 1
// CHECK-NEXT: |     |   `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:42> 'int' 1
// CHECK-NEXT: |     `-RecordType 0x{{[0-9A-Fa-f]+}} 'A<int, 0>'
// CHECK-NEXT: |       `-ClassTemplateSpecialization 0x{{[0-9A-Fa-f]+}} 'A'

template <class B1, B1 B2> using B = __make_integer_seq<A, B1, B2>;
using test2 = B<int, 1>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:20:1, col:23> col:7 test2 'B<int, 1>':'A<int, 0>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} 'B<int, 1>' sugar
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} 'B<int, 1>' sugar alias B
// CHECK-NEXT:       |-TemplateArgument type 'int'
// CHECK-NEXT:       | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:       |-TemplateArgument expr
// CHECK-NEXT:       | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <col:22> 'int'
// CHECK-NEXT:       |   |-value: Int 1
// CHECK-NEXT:       |   `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:22> 'int' 1
// CHECK-NEXT:       `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, 1>' sugar
// CHECK-NEXT:         `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, 1>' sugar __make_integer_seq
// CHECK-NEXT:           |-TemplateArgument template A
// CHECK-NEXT:           |-TemplateArgument type 'int':'int'
// CHECK-NEXT:           | `-SubstTemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'int' sugar
// CHECK-NEXT:           |   |-TemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'B1' dependent depth 0 index 0
// CHECK-NEXT:           |   | `-TemplateTypeParm 0x{{[0-9A-Fa-f]+}} 'B1'
// CHECK-NEXT:           |   `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:           |-TemplateArgument expr
// CHECK-NEXT:           | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <line:19:64> 'int'
// CHECK-NEXT:           |   |-value: Int 1
// CHECK-NEXT:           |   `-SubstNonTypeTemplateParmExpr 0x{{[0-9A-Fa-f]+}} <col:64> 'int'
// CHECK-NEXT:           |     |-NonTypeTemplateParmDecl 0x{{[0-9A-Fa-f]+}} <col:21, col:24> col:24 referenced 'B1' depth 0 index 1 B2
// CHECK-NEXT:           |     `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:64> 'int' 1
// CHECK-NEXT:           `-RecordType 0x{{[0-9A-Fa-f]+}} 'A<int, 0>'
// CHECK-NEXT:             `-ClassTemplateSpecialization 0x{{[0-9A-Fa-f]+}} 'A'

template <template <class T, T...> class S, class T, int N> struct C {
  using test3 = __make_integer_seq<S, T, N>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:48:3, col:43> col:9 test3 '__make_integer_seq<S, T, N>':'__make_integer_seq<S, T, N>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<S, T, N>' sugar dependent
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<S, T, N>' dependent __make_integer_seq
// CHECK-NEXT:       |-TemplateArgument template S
// CHECK-NEXT:       |-TemplateArgument type 'T'
// CHECK-NEXT:       | `-TemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'T' dependent depth 0 index 1
// CHECK-NEXT:       |   `-TemplateTypeParm 0x{{[0-9A-Fa-f]+}} 'T'
// CHECK-NEXT:       `-TemplateArgument expr
// CHECK-NEXT:         `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:42> 'type-parameter-0-1':'type-parameter-0-1' <Dependent>
// CHECK-NEXT:           `-DeclRefExpr 0x{{[0-9A-Fa-f]+}} <col:42> 'int' NonTypeTemplateParm 0x{{[0-9A-Fa-f]+}} 'N' 'int'

  using test4 = __make_integer_seq<A, T, 1>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:60:3, col:43> col:9 test4 '__make_integer_seq<A, T, 1>':'__make_integer_seq<A, T, 1>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, T, 1>' sugar dependent
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, T, 1>' dependent __make_integer_seq
// CHECK-NEXT:       |-TemplateArgument template A
// CHECK-NEXT:       |-TemplateArgument type 'T'
// CHECK-NEXT:       | `-TemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'T' dependent depth 0 index 1
// CHECK-NEXT:       |   `-TemplateTypeParm 0x{{[0-9A-Fa-f]+}} 'T'
// CHECK-NEXT:       `-TemplateArgument expr
// CHECK-NEXT:         `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:42> 'type-parameter-0-1':'type-parameter-0-1' <Dependent>
// CHECK-NEXT:           `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:42> 'int' 1

  using test5 = __make_integer_seq<A, int, N>;
//      CHECK: `-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:72:3, col:45> col:9 test5 '__make_integer_seq<A, int, N>':'__make_integer_seq<A, int, N>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, N>' sugar dependent
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__make_integer_seq<A, int, N>' dependent __make_integer_seq
// CHECK-NEXT:       |-TemplateArgument template A
// CHECK-NEXT:       |-TemplateArgument type 'int'
// CHECK-NEXT:       | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:       `-TemplateArgument expr
// CHECK-NEXT:         `-DeclRefExpr 0x{{[0-9A-Fa-f]+}} <col:44> 'int' NonTypeTemplateParm 0x{{[0-9A-Fa-f]+}} 'N' 'int'
};

template <class T, class S> struct D; // expected-note {{template is declared here}}
template <class T> struct D<T, __make_integer_seq<A, int, sizeof(T)>> {};
template struct D<char, A<int, 0>>; // expected-error {{explicit instantiation of undefined template}}

template <class T, class S> struct E; // expected-note {{template is declared here}}
template <class T> struct E<T, __make_integer_seq<A, T, 2>> {};
template struct E<short, A<short, 0, 1>>; // expected-error {{explicit instantiation of undefined template}}

template <template <class A1, A1... A2> class T, class S> struct F; // expected-note {{template is declared here}}
template <template <class A1, A1... A2> class T> struct F<T, __make_integer_seq<T, long, 3>> {};
template struct F<A, A<long, 0, 1, 2>>; // expected-error {{explicit instantiation of undefined template}}

template <class T> struct G;
template <class T> struct G<__make_integer_seq<A, T, 1>> {};
template <class T> struct G<__make_integer_seq<A, T, 1U>> {};

template <int S, class = __make_integer_seq<A, int, S>> struct H;
template <int S, int... Is> struct H<S, A<int, Is...>> { };

template <int S> void h(H<S>); // expected-note {{could not match '__make_integer_seq' against 'A'}}
void test_h() { h(H<5>{}); } // expected-error {{no matching function for call to 'h'}}
