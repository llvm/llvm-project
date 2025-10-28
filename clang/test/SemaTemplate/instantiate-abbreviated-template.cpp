// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify


template<typename...>
concept C = false; // expected-note 9{{because}}

template<typename T>
struct S {
    template<typename U>
    static void foo1(U a, auto b);
    static void foo2(T a, C<T> auto b);
    // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
    static void foo3(T a, C<decltype(a)> auto b);
    // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
    static void foo4(T a, C<decltype(a)> auto b, const C<decltype(b)> auto &&c);
    // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}}
};

using sf1 = decltype(S<int>::foo1(1, 2));
using sf2 = decltype(S<int>::foo2(1, 2)); // expected-error{{no matching function}}
using sf3 = decltype(S<int>::foo3(1, 2)); // expected-error{{no matching function}}
using sf4 = decltype(S<int>::foo4(1, 2, 3)); // expected-error{{no matching function}}


template<typename... T>
struct G {
    static void foo1(auto a, const C<decltype(a)> auto &&... b);
    // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}} expected-note@-1 3{{and}}
    static void foo2(auto a, const C<decltype(a), T> auto &&... b);
    // expected-note@-1{{candidate template ignored}} expected-note@-1{{because}} expected-note@-1{{and}}
};

using gf1 = decltype(G<int, char>::foo1('a', 1, 2, 3, 4)); // expected-error{{no matching function}}
using gf2 = decltype(G<int, char>::foo2('a', 1, 2)); // expected-error{{no matching function}}


// Regression (bug #45102): check that instantiation works where there is no
// TemplateTypeParmDecl
template <typename T> using id = T;

template <typename T>
constexpr void g() {
  id<void (T)> f;
}

static_assert((g<int>(), true));