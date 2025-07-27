// RUN:  %clang_cc1 -std=c++2b -verify %s

namespace Errors {

template <template<typename T> auto>
struct S1;
template <template<auto T> auto>
struct S2;
template <template<typename T> concept>
struct S3;
template <template<auto T> concept>
struct S4;
int a;

template <typename T>
concept C = true; // expected-note 2{{template argument refers to a concept 'C', here}}
template <typename T>
auto Var = 0; // expected-note 2{{template argument refers to a variable template 'Var', here}}

S1<1> t1; // expected-error {{template argument for template template parameter must be a variable template}}
S1<a> t2; // expected-error {{template argument for template template parameter must be a variable template}}
S1<int> t3; // expected-error {{template argument for template template parameter must be a variable template}}
S1<C> t4; // expected-error {{template argument does not refer to a variable template, or template template parameter}}
S2<1> t5; // expected-error {{template argument for template template parameter must be a variable template}}
S2<a> t6; // expected-error {{template argument for template template parameter must be a variable template}}
S2<int> t7; // expected-error {{template argument for template template parameter must be a variable template}}
S2<C> t8; // expected-error {{template argument does not refer to a variable template, or template template parameter}}
S3<1> t9; // expected-error {{template argument for template template parameter must be a concept}}
S3<a> t10; // expected-error {{template argument for template template parameter must be a concept}}
S3<int> t11; // expected-error {{template argument for template template parameter must be a concept}}
S3<Var> t12; // expected-error {{template argument does not refer to a concept, or template template parameter}}
S4<1> t13; // expected-error {{template argument for template template parameter must be a concept}}
S4<a> t14; // expected-error {{template argument for template template parameter must be a concept}}
S4<int> t15; // expected-error {{template argument for template template parameter must be a concept}}
S4<Var> t16; // expected-error {{template argument does not refer to a concept, or template template parameter}}

}

template <template<typename T> auto V> // expected-note {{previous template template parameter is here}}
struct S1 {
    static_assert(V<int> == 42);
    static_assert(V<const int> == 84);
    static_assert(V<double> == 0);
};
template <template<auto T> auto V>  // expected-note {{previous template template parameter is here}}
struct S2 {
    static_assert(V<0> == 1);
    static_assert(V<1> == 0);
};
template <template<typename T> concept C > // expected-note {{previous template template parameter is here}}
struct S3 {
    static_assert(C<int>);
};
template <template<auto> concept C> // expected-note {{previous template template parameter is here}}
struct S4 {
    static_assert(C<0>);
};

template <typename T> // expected-note {{template parameter has a different kind in template argument}}
concept C = true;

template <auto I> // expected-note {{template parameter has a different kind in template argument}}
concept CI = true;

template <typename T> // expected-note {{template parameter has a different kind in template argument}}
constexpr auto Var = 42;
template <typename T>
constexpr auto Var<const T> = 84;
template <>
constexpr auto Var<double> = 0;

template <auto N> // expected-note {{template parameter has a different kind in template argument}}
constexpr auto Var2 = 0;
template <auto N>
requires (N%2 == 0)
constexpr auto Var2<N> = 1;

void test () {
    S1<Var2> sE; // expected-error {{template template argument has different template parameters than its corresponding template template parameter}}
    S2<Var>  sE; // expected-error {{template template argument has different template parameters than its corresponding template template parameter}}
    S1<Var> s1;
    S2<Var2> s2;
    S3<C> s3;
    S4<C> sE; // expected-error {{template template argument has different template parameters than its corresponding template template parameter}}
    S4<CI> s4;
    S3<CI> sE; // expected-error {{template template argument has different template parameters than its corresponding template template parameter}}
}

namespace subsumption {

template <typename T>
concept A = true;

template <typename T>
concept B = A<T> && true;

template <typename T>
concept C = true;

template <typename T>
concept D = C<T> && true;

template <typename ABC, template <typename> concept... C>
concept Apply = (C<ABC> && ...);

constexpr int f(Apply<A, C> auto) {return 1;}
constexpr int f(Apply<B, D> auto) {return 2;}

int test() {
   static_assert(f(1) == 2);
}

}

namespace template_type_constraints {


template <typename T>
concept Unary = true;
template <typename T, typename = int>
concept BinaryDefaulted = true;

template <typename T>
concept UnaryFalse = false; // expected-note 3{{because 'false' evaluated to false}}
template <typename T, typename = int>
concept BinaryDefaultedFalse = false;

template <template <typename...> concept C, typename T>
struct S {
    template <C TT> // expected-note {{because 'int' does not satisfy 'UnaryFalse'}}
    void f(TT); // expected-note {{ignored}}
    void g(C auto); // expected-note {{ignored}} \
                    // expected-note {{because 'int' does not satisfy 'UnaryFalse'}}

    auto h() -> C auto { // expected-error {{deduced type 'int' does not satisfy 'UnaryFalse'}}
        return 0;
    };

    void test() {
        C auto a = 0;
    }
};

template <template <typename...> concept C, typename T>
struct SArg {
    template <C<int> TT>
    void f(TT);
    void g(C<int> auto);

    auto h() -> C<int> auto {
        return 0;
    };
    void test() {
        C<int> auto a = 0;
    }
};

void test() {
    S<Unary, int> s;
    s.f(0);
    s.g(0);
    s.h();
    S<BinaryDefaulted, int> s2;
    s2.f(0);
    s2.g(0);
    s2.h();

    SArg<BinaryDefaulted, int> s3;
    s3.f(0);
    s3.g(0);
    s3.h();
}

void test_errors() {
    S<UnaryFalse, int> s;
    s.f(0); // expected-error {{no matching member function for call to 'f'}}
    s.g(0); // expected-error {{no matching member function for call to 'g'}}
    s.h(); // expected-note {{in instantiation of member function 'template_type_constraints::S<template_type_constraints::UnaryFalse, int>::h'}}
}

}

template <typename T>
concept Unary = true;
template <typename T, typename = int>
concept BinaryDefaulted = true;

template <typename T>
concept UnaryFalse = false; // expected-note 3{{because 'false' evaluated to false}}
template <typename T, typename = int>
concept BinaryDefaultedFalse = false;

template <template <typename...> concept C, typename T>
struct S {
    template <C TT> // expected-note {{because 'int' does not satisfy 'UnaryFalse'}}
    void f(TT); // expected-note {{ignored}}
    void g(C auto); // expected-note {{ignored}} \
                    // expected-note {{because 'int' does not satisfy 'UnaryFalse'}}

    auto h() -> C auto { // expected-error {{deduced type 'int' does not satisfy 'UnaryFalse'}}
        return 0;
    };

    void test() {
        C auto a = 0;
    }
};

template <template <typename...> concept C, typename T>
struct SArg {
    template <C<int> TT>
    void f(TT);
    void g(C<int> auto);

    auto h() -> C<int> auto {
        return 0;
    };
    void test() {
        C<int> auto a = 0;
    }
};

void test_args() {
    S<Unary, int> s;
    s.f(0);
    s.g(0);
    s.h();
    S<BinaryDefaulted, int> s2;
    s2.f(0);
    s2.g(0);
    s2.h();

    SArg<BinaryDefaulted, int> s3;
    s3.f(0);
    s3.g(0);
    s3.h();
}

void test_errors() {
    S<UnaryFalse, int> s;
    s.f(0); // expected-error {{no matching member function for call to 'f'}}
    s.g(0); // expected-error {{no matching member function for call to 'g'}}
    s.h(); // expected-note {{in instantiation of member function 'S<UnaryFalse, int>::h'}}
}

namespace non_type {

template <auto>
concept Unary = true;

template <template <auto> concept C>
struct S {
    template <C Foo> // expected-error {{concept named in type constraint is not a type concept}}
    void f();
    // FIXME, bad diagnostic
    void g(C auto);  // expected-error{{concept named in type constraint is not a type concept}}
    auto h() -> C auto {  // expected-error{{concept named in type constraint is not a type concept}}
    }
    void i() {
        C auto a = 0;  // expected-error{{concept named in type constraint is not a type concept}}
    }
};

}

namespace default_args {

template <typename T>
concept Concept = false; // expected-note 2{{template argument refers to a concept 'Concept', here}} \
                         // expected-note 2{{because 'false' evaluated to false}}

template <typename T>
constexpr auto Var = false; // expected-note 2{{template argument refers to a variable template 'Var', here}}

template <typename T>
struct Type; // expected-note 2{{template argument refers to a class template 'Type', here}}


template <template <typename> auto = Concept> // expected-error {{template argument does not refer to a variable template, or template template parameter}}
struct E1;

template <template <typename> auto  = Type> // expected-error {{template argument does not refer to a variable template, or template template parameter}}
struct E2;

template <template <typename> typename = Concept> // expected-error {{template argument does not refer to a class or alias template, or template template parameter}}
struct E3;

template <template <typename> typename  = Var> // expected-error {{template argument does not refer to a class or alias template, or template template parameter}}
struct E4;

template <template <typename> concept  = Var> // expected-error {{template argument does not refer to a concept, or template template parameter}}
struct E4;

template <template <typename> concept  = Type> // expected-error {{template argument does not refer to a concept, or template template parameter}}
struct E4;

template <
    template <typename> concept TConcept, // expected-note 2{{template argument refers to a concept 'TConcept', here}}
    template <typename> auto TVar, // expected-note 2{{template argument refers to a variable template 'TVar', here}}
    template <typename> typename TType // expected-note 2{{template argument refers to a class template 'TType', here}}
>
struct Nested {
    template <template <typename> auto = TConcept> // expected-error {{template argument does not refer to a variable template, or template template parameter}}
    struct E1;

    template <template <typename> auto  = TType> // expected-error {{template argument does not refer to a variable template, or template template parameter}}
    struct E2;

    template <template <typename> typename = TConcept> // expected-error {{template argument does not refer to a class or alias template, or template template parameter}}
    struct E3;

    template <template <typename> typename  = TVar> // expected-error {{template argument does not refer to a class or alias template, or template template parameter}}
    struct E4;

    template <template <typename> concept  = TVar> // expected-error {{template argument does not refer to a concept, or template template parameter}}
    struct E4;

    template <template <typename> concept  = TType> // expected-error {{template argument does not refer to a concept, or template template parameter}}
    struct E4;
};


template <template <typename> concept C = Concept>
struct TestDefaultConcept {
    template <template <typename> concept CC = C>
    void f() {
        static_assert(C<int>); // expected-error {{static assertion failed}} \
                               // expected-note {{because 'int' does not satisfy 'Concept'}}
        static_assert(CC<int>);  // expected-error {{static assertion failed}} \
                                 // expected-note {{because 'int' does not satisfy 'Concept'}}
    }
};
void do_test_concept() {
    TestDefaultConcept<>{}.f(); // expected-note {{in instantiation}}
}

template <template <typename> auto V = Var>
struct TestDefaultVar {
    template <template <typename> auto VV = V>
    void f() {
        static_assert(V<int>); // expected-error {{static assertion failed}}
        static_assert(VV<int>); // expected-error {{static assertion failed}}
    }
};
void do_test_var() {
    TestDefaultVar<>{}.f(); // expected-note {{in instantiation}}
}

}

namespace TTPDependence {
template <template <typename... > concept C>
concept A = C<>;
template <template <typename... > concept C>
concept B = C<int>;

template <template <typename... > auto Var>
concept C = Var<>;
template <template <typename... > auto Var>
concept D = Var<int>;

}

namespace InvalidName {
// FIXME corentin: improve diagnostics
template <typename T, template <typename> concept C>
concept A = C<T>; // expected-note {{here}}

template <A<concept missing<int>> T> // expected-error {{expected expression}} \
                                     // expected-error {{too few template arguments for concept 'A'}}
auto f();
}
