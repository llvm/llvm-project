// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

static_assert(__has_builtin(__make_integer_seq), "");

template <class T, T... I>
struct Seq {
  static constexpr T PackSize = sizeof...(I);
};

template <typename T, T N>
using MakeSeq = __make_integer_seq<Seq, T, N>;

static_assert(__is_same(MakeSeq<int, 0>, Seq<int>), "");
static_assert(__is_same(MakeSeq<int, 1>, Seq<int, 0>), "");
static_assert(__is_same(MakeSeq<int, 2>, Seq<int, 0, 1>), "");
static_assert(__is_same(MakeSeq<int, 3>, Seq<int, 0, 1, 2>), "");
static_assert(__is_same(MakeSeq<int, 4>, Seq<int, 0, 1, 2, 3>), "");

static_assert(__is_same(MakeSeq<unsigned int, 0U>, Seq<unsigned int>), "");
static_assert(__is_same(MakeSeq<unsigned int, 1U>, Seq<unsigned int, 0U>), "");
static_assert(__is_same(MakeSeq<unsigned int, 2U>, Seq<unsigned int, 0U, 1U>), "");
static_assert(__is_same(MakeSeq<unsigned int, 3U>, Seq<unsigned int, 0U, 1U, 2U>), "");
static_assert(__is_same(MakeSeq<unsigned int, 4U>, Seq<unsigned int, 0U, 1U, 2U, 3U>), "");

static_assert(__is_same(MakeSeq<long long, 0LL>, Seq<long long>), "");
static_assert(__is_same(MakeSeq<long long, 1LL>, Seq<long long, 0LL>), "");
static_assert(__is_same(MakeSeq<long long, 2LL>, Seq<long long, 0LL, 1LL>), "");
static_assert(__is_same(MakeSeq<long long, 3LL>, Seq<long long, 0LL, 1LL, 2LL>), "");
static_assert(__is_same(MakeSeq<long long, 4LL>, Seq<long long, 0LL, 1LL, 2LL, 3LL>), "");

static_assert(__is_same(MakeSeq<unsigned long long, 0ULL>, Seq<unsigned long long>), "");
static_assert(__is_same(MakeSeq<unsigned long long, 1ULL>, Seq<unsigned long long, 0ULL>), "");
static_assert(__is_same(MakeSeq<unsigned long long, 2ULL>, Seq<unsigned long long, 0ULL, 1ULL>), "");
static_assert(__is_same(MakeSeq<unsigned long long, 3ULL>, Seq<unsigned long long, 0ULL, 1ULL, 2ULL>), "");
static_assert(__is_same(MakeSeq<unsigned long long, 4ULL>, Seq<unsigned long long, 0ULL, 1ULL, 2ULL, 3ULL>), "");

template <typename T, T N>
using ErrorSeq = __make_integer_seq<Seq, T, N>; // expected-error{{must have non-negative sequence length}} \
                                                   expected-error{{must have integral element type}}

enum Color : int { Red,
                   Green,
                   Blue };
using illformed1 = ErrorSeq<Color, Blue>; // expected-note{{in instantiation}}

using illformed2 = ErrorSeq<int, -5>; // expected-note{{in instantiation}}

template <typename T, T N> void f() {}
__make_integer_seq<f, int, 0> x; // expected-error{{template template parameter must be a class template or type alias template}}

__make_integer_seq<__make_integer_seq, int, 10> PR28494; // expected-note{{different template parameters}}
// expected-error@make_integer_seq.cpp:* {{template argument for template template parameter must be a class template or type alias template}}
