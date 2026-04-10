// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++2c -verify %s


namespace Ex1 {
    template<typename T>
    struct C {
    C(T);
    };
    template<template<typename> class X>
    void f() {
    X x(1);
    }
    template void f<C>();
}

namespace Ex2 {
template<typename ... T>
struct C {
    C(T ...);
};
template<template<typename> class X>
void f() {
    X x1{1};
    X x2{1, 2};  // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'Ex2::C'}} \
    // expected-note {{candidate function template not viable: requires 1 argument, but 2 were provided}} \
    // expected-note {{implicit deduction guide declared as 'template <typename> requires __is_deducible(Ex2::(anonymous), Ex2::C<type-parameter-0-0>) (type-parameter-0-0) -> Ex2::C<type-parameter-0-0>'}} \
    // expected-note {{candidate function template not viable: requires 1 argument, but 2 were provided}} \
    // expected-note {{implicit deduction guide declared as 'template <typename> requires __is_deducible(Ex2::(anonymous), Ex2::C<type-parameter-0-0>) (Ex2::C<type-parameter-0-0>) -> Ex2::C<type-parameter-0-0}}
}
template void f<C>(); // expected-note {{in instantiation}}
}

namespace Ex3 {
    template<typename T = int>
    struct C {
    C(int);
    };
    template<template<typename = long> class X>
    void f() {
    X x(1);
    }
    template void f<C>();
}

namespace Ex4 {
template<int>
struct A { };
template<int I>
struct C {
    C(A<I>);
};
template<template<short> class X>
void f() {
    X x1{A<1>()}; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'Ex4::C'}} \
                  // expected-note  {{candidate template ignored: substitution failure: deduced non-type template argument does not have the same type as the corresponding template parameter ('int' vs 'short')}} \
                  // expected-note  {{implicit deduction guide declared as 'template <short> requires __is_deducible(Ex4::(anonymous), Ex4::C<value-parameter-0-0>) (A<value-parameter-0-0>) -> Ex4::C<value-parameter-0-0>'}} \
                  // expected-note  {{candidate template ignored: could not match 'Ex4::C' against 'A'}} \
                  // expected-note  {{implicit deduction guide declared as 'template <short> requires __is_deducible(Ex4::(anonymous), Ex4::C<value-parameter-0-0>) (Ex4::C<value-parameter-0-0>) -> Ex4::C<value-parameter-0-0>'}}
}
template void f<C>(); // expected-note {{in instantiation}}
}


namespace CWG3003 {

template <typename T> struct A { A(T); };

template <typename T, template <typename> class TT = A>
using Alias = TT<T>; // expected-note {{template is declared here}}

template <typename T>
using Alias2 = Alias<T>;

void h() { Alias2 a(42); } // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'Alias2'}}
void h2() { Alias a(42); } // expected-error {{alias template 'Alias' requires template arguments; argument deduction only allowed for class templates or alias templates}}

}


namespace WordingExample {
template<typename ... Ts>
struct Y {
    Y();
    Y(Ts ...);
};
template<template<typename T = char> class X>
void f() {
    X x0{};
    X x1{1};
    X x2{1, 2}; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'WordingExample::Y'}} \
                // expected-note {{candidate function template not viable: requires 1 argument, but 2 were provided}} \
                // expected-note {{implicit deduction guide declared as 'template <typename T = char> requires __is_deducible(WordingExample::(anonymous), WordingExample::Y<T>) (T) -> WordingExample::Y<T>'}} \
                // expected-note {{candidate function template not viable: requires 1 argument, but 2 were provided}} \
                // expected-note {{implicit deduction guide declared as 'template <typename T = char> requires __is_deducible(WordingExample::(anonymous), WordingExample::Y<T>) (WordingExample::Y<T>) -> WordingExample::Y<T>'}} \
                // expected-note {{candidate function template not viable: requires 0 arguments, but 2 were provided}} \
                // expected-note {{implicit deduction guide declared as 'template <typename T = char> requires __is_deducible(WordingExample::(anonymous), WordingExample::Y<T>) () -> WordingExample::Y<T>'}}
};
template void f<Y>(); // expected-note {{in instantiation}}

}
