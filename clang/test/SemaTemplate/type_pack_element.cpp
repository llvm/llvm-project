// RUN: %clang_cc1 -std=c++2b -fsyntax-only -triple x86_64-linux-gnu -ast-dump -verify -xc++ < %s | FileCheck %s

using test1 = __type_pack_element<0, int>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <<stdin>:3:1, col:41> col:7 test1 '__type_pack_element<0, int>':'int'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__type_pack_element<0, int>' sugar
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__type_pack_element<0, int>' sugar __type_pack_element
// CHECK-NEXT:       |-TemplateArgument expr
// CHECK-NEXT:       | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <col:35> 'unsigned long'
// CHECK-NEXT:       |   |-value: Int 0
// CHECK-NEXT:       |   `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:35> 'unsigned long' <IntegralCast>
// CHECK-NEXT:       |     `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:35> 'int' 0
// CHECK-NEXT:       |-TemplateArgument type 'int'
// CHECK-NEXT:       | `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
// CHECK-NEXT:       `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'

template<int N, class ...Ts> struct A {
  using test2 = __type_pack_element<N, Ts...>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:17:3, col:45> col:9 test2 '__type_pack_element<N, Ts...>':'__type_pack_element<N, Ts...>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__type_pack_element<N, Ts...>' sugar dependent
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__type_pack_element<N, Ts...>' dependent __type_pack_element
// CHECK-NEXT:       |-TemplateArgument expr
// CHECK-NEXT:       | `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:37> 'unsigned long' <IntegralCast>
// CHECK-NEXT:       |   `-DeclRefExpr 0x{{[0-9A-Fa-f]+}} <col:37> 'int' NonTypeTemplateParm 0x{{[0-9A-Fa-f]+}} 'N' 'int'
// CHECK-NEXT:       `-TemplateArgument type 'Ts...'
// CHECK-NEXT:         `-PackExpansionType 0x{{[0-9A-Fa-f]+}} 'Ts...' dependent
// CHECK-NEXT:           `-TemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'Ts' dependent contains_unexpanded_pack depth 0 index 1 pack
// CHECK-NEXT:             `-TemplateTypeParm 0x{{[0-9A-Fa-f]+}} 'Ts'

  using test3 = __type_pack_element<0, Ts...>;
//      CHECK: |-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:29:3, col:45> col:9 test3 '__type_pack_element<0, Ts...>':'__type_pack_element<0, Ts...>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__type_pack_element<0, Ts...>' sugar dependent
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__type_pack_element<0, Ts...>' dependent __type_pack_element
// CHECK-NEXT:       |-TemplateArgument expr
// CHECK-NEXT:       | `-ConstantExpr 0x{{[0-9A-Fa-f]+}} <col:37> 'unsigned long'
// CHECK-NEXT:       |   |-value: Int 0
// CHECK-NEXT:       |   `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:37> 'unsigned long' <IntegralCast>
// CHECK-NEXT:       |     `-IntegerLiteral 0x{{[0-9A-Fa-f]+}} <col:37> 'int' 0
// CHECK-NEXT:       `-TemplateArgument type 'Ts...'
// CHECK-NEXT:         `-PackExpansionType 0x{{[0-9A-Fa-f]+}} 'Ts...' dependent
// CHECK-NEXT:           `-TemplateTypeParmType 0x{{[0-9A-Fa-f]+}} 'Ts' dependent contains_unexpanded_pack depth 0 index 1 pack
// CHECK-NEXT:             `-TemplateTypeParm 0x{{[0-9A-Fa-f]+}} 'Ts'

  using test4 = __type_pack_element<N, int>;
//      CHECK: `-TypeAliasDecl 0x{{[0-9A-Fa-f]+}} <line:43:3, col:43> col:9 test4 '__type_pack_element<N, int>':'__type_pack_element<N, int>'
// CHECK-NEXT:   `-ElaboratedType 0x{{[0-9A-Fa-f]+}} '__type_pack_element<N, int>' sugar dependent
// CHECK-NEXT:     `-TemplateSpecializationType 0x{{[0-9A-Fa-f]+}} '__type_pack_element<N, int>' dependent __type_pack_element
// CHECK-NEXT:       |-TemplateArgument expr
// CHECK-NEXT:       | `-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:37> 'unsigned long' <IntegralCast>
// CHECK-NEXT:       |   `-DeclRefExpr 0x{{[0-9A-Fa-f]+}} <col:37> 'int' NonTypeTemplateParm 0x{{[0-9A-Fa-f]+}} 'N' 'int'
// CHECK-NEXT:       `-TemplateArgument type 'int'
// CHECK-NEXT:         `-BuiltinType 0x{{[0-9A-Fa-f]+}} 'int'
};

template <class T, class S> struct B; // expected-note {{template is declared here}}
template <class T> struct B<T, __type_pack_element<sizeof(T), void, long>> {};
template struct B<char, long>; // expected-error {{explicit instantiation of undefined template}}

template <class T, class S> struct C; // expected-note {{template is declared here}}
template <class T> struct C<T, __type_pack_element<0, T, short>> {};
template struct C<int, int>; // expected-error {{explicit instantiation of undefined template}}

template <class T> struct D;
template <class T, class U> struct D<__type_pack_element<0, T, U>> {};
template <class T, class U> struct D<__type_pack_element<0, U, T>> {};

template <class T> struct E;
template <class T> struct E<__type_pack_element<0, T>> {};
template <class T, class U> struct E<__type_pack_element<0, T, U>> {};
