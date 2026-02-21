// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-error@* 2 {{template argument for non-type template parameter must be an expression}}

using SizeT = decltype(sizeof(int));

// Dependent cases that previously crashed but now return QualType() gracefully.
template <SizeT... Seq> // expected-note {{template parameter is declared here}}
using gh180307 = __type_pack_element<Seq...>;

template <typename T>
using gh180307_bis = __make_integer_seq<gh180307, T>;
// expected-note@-1 {{template template argument has different template parameters than its corresponding template template parameter}}

// Eager expansion checks: Built-in templates should expand even if the
// destination template OR the type argument is dependent, provided the size is known.
template <template <typename T, T... Ints> class Seq>
using test_make_integer_seq_eager = __make_integer_seq<Seq, int, 2>;

template <typename T, T... Ints> struct MySeq;
using check_eager = test_make_integer_seq_eager<MySeq>;
using check_eager = MySeq<int, 0, 1>;

template <typename T>
using test_make_integer_seq_type_dependent = __make_integer_seq<MySeq, T, 2>;
using check_type_eager = test_make_integer_seq_type_dependent<int>;
using check_type_eager = MySeq<int, 0, 1>;

// Too many arguments tests
template <int N> struct S; // expected-note {{template parameter is declared here}}
using too_many_args = __make_integer_seq<S, int, 10, int>; 
// expected-note@-1 {{template template argument has different template parameters than its corresponding template template parameter}}

// Too few arguments tests
template <SizeT Index>
using too_few_args = __type_pack_element<Index>;

// Verify that too_few_args doesn't crash on instantiation either
// (It should just be an invalid type)
template <SizeT I> struct Wrap {
  using type = too_few_args<I>;
};
