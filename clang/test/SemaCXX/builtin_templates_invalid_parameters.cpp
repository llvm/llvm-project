// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-note@* 0+ {{from hidden source}}

using SizeT = decltype(sizeof(int));

// 1. Check pack expansion into non-pack parameter
template <SizeT... Seq>
using error_expansion = __type_pack_element<Seq...>; 
// expected-error@-1 {{pack expansion used as argument for non-pack parameter of builtin template}}

// 2. Check Arity mismatch (Too many/few)
template <int N> struct S; // expected-note 1+ {{template parameter is declared here}}
using too_many_args = __make_integer_seq<S, int, 2, int>;
// expected-error@* {{template argument for non-type template parameter must be an expression}}
// expected-note@* {{template template argument has different template parameters}}

using too_few_args = __type_pack_element<>;
// expected-error@-1 {{too few template arguments for template '__type_pack_element'}}

// Verify that the compiler survives even if the alias refers back to an invalid one
// (We expect an 'undeclared identifier' here because error_expansion failed above)
template <typename T>
using bis = __make_integer_seq<error_expansion, T>;
// expected-error@-1 {{use of undeclared identifier 'error_expansion'}}
