// RUN: %clang_cc1 -std=c++2c -verify %s

template <class T> concept A = true;
template <class T> concept C = A<T> && true;
template <class T> concept D = A<T> && __is_same(T, int);


template <class T> requires (A<T>)
constexpr int f(T) { return 0; };
template <class... T> requires (C<T> && ...)
constexpr int f(T...) { return 1; };

static_assert(f(0) == 0);
static_assert(f(1) == 0);


template <class... T> requires (A<T> && ...)
constexpr int g(T...) { return 0; };
template <class... T> requires (C<T> && ...)
constexpr int g(T...) { return 1; };

static_assert(g(0) == 1);
static_assert(g() == 1);
static_assert(g(1, 2) == 1);



template <class... T> requires (A<T> && ...)
constexpr int h(T...) { return 0; }; // expected-note {{candidate}}
template <class... T> requires (C<T> || ...)
constexpr int h(T...) { return 1; }; // expected-note {{candidate}}

static_assert(h(0) == 1); // expected-error {{call to 'h' is ambiguous}}

template <class... T> requires (A<T> || ...)
constexpr int i(T...) { return 0; }; // expected-note {{candidate}}
template <class... T> requires (C<T> && ...)
constexpr int i(T...) { return 1; }; // expected-note {{candidate}}

static_assert(i(0) == 1); // expected-error {{call to 'i' is ambiguous}}


template <class... T> requires (A<T> || ... || true)
constexpr int j(T...) { return 0; };
template <class... T> requires (C<T> && ... && true)
constexpr int j(T...) { return 1; };

static_assert(j(0) == 1);
static_assert(j() == 1);



template <class... T> requires (A<T> || ...)
constexpr int k(T...) { return 0; }; // expected-note {{candidate template ignored: constraints not satisfied [with T = <>]}}
template <class... T> requires (C<T> || ...)
constexpr int k(T...) { return 1; }; // expected-note {{candidate template ignored: constraints not satisfied [with T = <>]}}

static_assert(k(0) == 1);
static_assert(k() == 0); // expected-error {{no matching function for call to 'k'}}
static_assert(k(1, 2) == 1);


consteval int terse(A auto...) {return 1;}
consteval int terse(D auto...) {return 2;}

static_assert(terse() == 2);
static_assert(terse(0, 0) == 2);
static_assert(terse(0L, 0) == 1);

template <A... T>
consteval int tpl_head(A auto...) {return 1;}
template <D... T>
consteval int tpl_head(D auto...) {return 2;}

static_assert(tpl_head() == 2);
static_assert(tpl_head(0, 0) == 2);
static_assert(tpl_head(0L, 0) == 1);


namespace equivalence {

template <typename... T>
struct S {
    template <typename... U>
    void f() requires (A<U> && ...);
    template <typename... U>
    void f() requires (C<U> && ...);

    template <typename... U>
    void g() requires (A<T> && ...);
    template <typename... U>
    void g() requires (C<T> && ...);

    template <typename... U>
    void h() requires (A<U> && ...); // expected-note {{candidate}}
    template <typename... U>
    void h() requires (C<T> && ...); // expected-note {{candidate}}
};

void test() {
    S<int>{}.f<int>();
    S<int>{}.g<int>();
    S<int>{}.h<int>(); // expected-error {{call to member function 'h' is ambiguous}}
}


}

namespace substitution {
    struct S {
    using type = int;
};

template <typename... T>
consteval int And1() requires (C<typename T::type> && ...) { // #and1
    return 1;
}

template <typename T, typename... U>
consteval int And2() requires (C<typename U::type> && ... && C<typename T::type>) { // #and2
    return 2;
}

template <typename T, typename... U>
consteval int And3() requires (C<typename T::type> && ... && C<typename U::type>) { // #and3
    return 3;
}

template <typename... T>
consteval int Or1() requires (C<typename T::type> || ...) { // #or1
    return 1;
}

template <typename T, typename... U>
consteval int Or2() requires (C<typename U::type> || ... || C<typename T::type>) {  // #or2
    return 2;
}

template <typename T, typename... U>
consteval int Or3() requires (C<typename T::type> || ... || C<typename U::type>) {  // #or3
    return 3;
}

static_assert(And1<>() == 1);
static_assert(And1<S>() == 1);
static_assert(And1<S, S>() == 1);
static_assert(And1<int>() == 1); // expected-error {{no matching function for call to 'And1'}}
                                 // expected-note@#and1 {{candidate template ignored: constraints not satisfied}}
                                 // expected-note@#and1 {{because substituted constraint expression is ill-formed}}

static_assert(And1<S, int>() == 1); // expected-error {{no matching function for call to 'And1'}}
                                   // expected-note@#and1 {{candidate template ignored: constraints not satisfied}}
                                   // expected-note@#and1 {{because substituted constraint expression is ill-formed}}

static_assert(And1<int, S>() == 1); // expected-error {{no matching function for call to 'And1'}}
                                   // expected-note@#and1 {{candidate template ignored: constraints not satisfied}}
                                   // expected-note@#and1 {{because substituted constraint expression is ill-formed}}

static_assert(And2<S>() == 2);
static_assert(And2<S, S>() == 2);
static_assert(And2<int>() == 2);

static_assert(And2<int, int>() == 2);  // expected-error {{no matching function for call to 'And2'}}
                                      // expected-note@#and2 {{candidate template ignored: constraints not satisfied}}
                                     // expected-note@#and2 {{because substituted constraint expression is ill-formed}}

static_assert(And2<S, int>() == 2); // expected-error {{no matching function for call to 'And2'}}
                                   // expected-note@#and2 {{candidate template ignored: constraints not satisfied}}
                                   // expected-note@#and2 {{because substituted constraint expression is ill-formed}}

static_assert(And2<int, S>() == 2); // expected-error {{no matching function for call to 'And2'}}
                                   // expected-note@#and2 {{candidate template ignored: constraints not satisfied}}
                                   // expected-note@#and2 {{because substituted constraint expression is ill-formed}}

static_assert(And3<S>() == 3);
static_assert(And3<S, S>() == 3);
static_assert(And3<int>() == 3);   // expected-error {{no matching function for call to 'And3'}}
                                   // expected-note@#and3 {{candidate template ignored: constraints not satisfied}}
                                   // expected-note@#and3 {{because substituted constraint expression is ill-formed}}

static_assert(And3<int, int>() == 3);  // expected-error {{no matching function for call to 'And3'}}
                                      // expected-note@#and3 {{candidate template ignored: constraints not satisfied}}
                                     // expected-note@#and3 {{because substituted constraint expression is ill-formed}}

static_assert(And3<S, int>() == 3); // expected-error {{no matching function for call to 'And3'}}
                                   // expected-note@#and3 {{candidate template ignored: constraints not satisfied}}
                                   // expected-note@#and3 {{because substituted constraint expression is ill-formed}}

static_assert(And3<int, S>() == 3); // expected-error {{no matching function for call to 'And3'}}
                                   // expected-note@#and3 {{candidate template ignored: constraints not satisfied}}
                                   // expected-note@#and3 {{because substituted constraint expression is ill-formed}}


static_assert(Or1<>() == 1); // expected-error {{no matching function for call to 'Or1'}}
                             // expected-note@#or1 {{candidate template ignored: constraints not satisfied}}
static_assert(Or1<S>() == 1);
static_assert(Or1<int, S>() == 1);
static_assert(Or1<S, int>() == 1);
static_assert(Or1<S, S>() == 1);
static_assert(Or1<int>() == 1); // expected-error {{no matching function for call to 'Or1'}}
                                // expected-note@#or1 {{candidate template ignored: constraints not satisfied}} \
                                // expected-note@#or1 {{because substituted constraint expression is ill-formed}}


static_assert(Or2<S>() == 2);
static_assert(Or2<int, S>() == 2);
static_assert(Or2<S, int>() == 2);
static_assert(Or2<S, S>() == 2);
static_assert(Or2<int>() == 2); // expected-error {{no matching function for call to 'Or2'}}
                                // expected-note@#or2 {{candidate template ignored: constraints not satisfied}} \
                                // expected-note@#or2 {{because substituted constraint expression is ill-formed}}

static_assert(Or3<S>() == 3);
static_assert(Or3<int, S>() == 3);
static_assert(Or3<S, int>() == 3);
static_assert(Or3<S, S>() == 3);
static_assert(Or3<int>() == 3); // expected-error {{no matching function for call to 'Or3'}}
                                // expected-note@#or3 {{candidate template ignored: constraints not satisfied}} \
                                // expected-note@#or3 {{because substituted constraint expression is ill-formed}}
}

namespace bool_conversion_break {

template <typename ...V> struct A;
struct Thingy {
    static constexpr int compare(const Thingy&) {return 1;}
};
template <typename ...T, typename ...U>
void f(A<T ...> *, A<U ...> *) // expected-note {{candidate template ignored: failed template argument deduction}}
requires (T::compare(U{}) && ...); // expected-error {{atomic constraint must be of type 'bool' (found 'int')}}

void g() {
    A<Thingy, Thingy> *ap;
    f(ap, ap); // expected-error{{no matching function for call to 'f'}} \
               // expected-note {{while checking constraint satisfaction}} \
               // expected-note {{in instantiation of function template specialization}}
}

}

namespace nested {

template <typename... T>
struct S {
    template <typename... U>
    consteval static int f()
        requires ((A<T> && ...) && ... && A<U> ) {
            return 1;
    }

    template <typename... U>
    consteval static int f()
        requires ((C<T> && ...) && ... && C<U> ) {
            return 2;
    }

    template <typename... U>
    consteval static int g() // #nested-ambiguous-g1
        requires ((A<T> && ...) && ... && A<U> ) {
            return 1;
    }

    template <typename... U>
    consteval static int g() // #nested-ambiguous-g2
        requires ((C<U> && ...) && ... && C<T> ) {
            return 2;
    }
};

static_assert(S<int>::f<int>() == 2);

static_assert(S<int>::g<int>() == 2); // expected-error {{call to 'g' is ambiguous}}
                                      // expected-note@#nested-ambiguous-g1 {{candidate}}
                                      // expected-note@#nested-ambiguous-g2 {{candidate}}


}

namespace GH99430 {

template <class _Ty1, class _Ty2>
using _Synth_three_way_result = int;

template <class... _Types>
class tuple;

template <int _Index>
struct tuple_element;

template <class, int...>
struct _Three_way_comparison_result_with_tuple_like {
  using type = int;
};

template <class... _TTypes, int... _Indices>
  requires(requires {
    typename _Synth_three_way_result<_TTypes, tuple_element<_Indices>>;
  } && ...)

struct _Three_way_comparison_result_with_tuple_like<tuple<_TTypes...>, _Indices...>{
    using type = long;
};

static_assert(__is_same_as(_Three_way_comparison_result_with_tuple_like<tuple<int>, 0, 1>::type, int));
static_assert(__is_same_as(_Three_way_comparison_result_with_tuple_like<tuple<int>, 0>::type, long));

}

namespace GH88866 {

template <typename...Ts> struct index_by;

template <typename T, typename Indices>
concept InitFunc = true;

namespace ExpandsBoth {

template <typename Indices, InitFunc<Indices> auto... init>
struct LazyLitMatrix; // expected-note {{here}}

template <
    typename...Indices,
    InitFunc<index_by<Indices>> auto... init
>
struct LazyLitMatrix<index_by<Indices...>, init...> {
};

// FIXME: Explain why we didn't pick up the partial specialization - pack sizes don't match.
template struct LazyLitMatrix<index_by<int, char>, 42>;
// expected-error@-1 {{instantiation of undefined template}}
template struct LazyLitMatrix<index_by<int, char>, 42, 43>;

}

namespace ExpandsRespectively {

template <typename Indices, InitFunc<Indices> auto... init>
struct LazyLitMatrix;

template <
    typename...Indices,
    InitFunc<index_by<Indices...>> auto... init
>
struct LazyLitMatrix<index_by<Indices...>, init...> {
};

template struct LazyLitMatrix<index_by<int, char>, 42>;
template struct LazyLitMatrix<index_by<int, char>, 42, 43>;

}

namespace TypeParameter {

template <typename Indices, InitFunc<Indices>... init>
struct LazyLitMatrix; // expected-note {{here}}

template <
    typename...Indices,
    InitFunc<index_by<Indices>>... init
>
struct LazyLitMatrix<index_by<Indices...>, init...> {
};

// FIXME: Explain why we didn't pick up the partial specialization - pack sizes don't match.
template struct LazyLitMatrix<index_by<int, char>, float>;
// expected-error@-1 {{instantiation of undefined template}}
template struct LazyLitMatrix<index_by<int, char>, unsigned, float>;

}

namespace Invalid {

template <typename Indices, InitFunc<Indices>... init>
struct LazyLitMatrix;

template <
    typename...Indices,
    InitFunc<index_by<Indices>> init
    // expected-error@-1 {{unexpanded parameter pack 'Indices'}}
>
struct LazyLitMatrix<index_by<Indices...>, init> {
};

}

}
