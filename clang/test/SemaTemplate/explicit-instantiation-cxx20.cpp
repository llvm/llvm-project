// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20  %s

template<class T>
concept C = true;

template<class T>
concept D = C<T> && true;

template<typename T>
struct a {
    void no_candidate() requires(false) {}
    // expected-note@-1 {{candidate function not viable: constraints not satisfied}} \
    // expected-note@-1 {{because 'false' evaluated to false}}
    void no_candidate() requires(false && false) {}
    // expected-note@-1 {{candidate function not viable: constraints not satisfied}} \
    // expected-note@-1 {{because 'false' evaluated to false}}

    void subsumes();
    void subsumes() requires C<T>;
    void subsumes() requires D<T> {};

    void ok() requires false;
    void ok() requires true {};

    void ok2() requires false;
    void ok2(){};

    void ambiguous() requires true;
    // expected-note@-1 {{candidate function}}
    void ambiguous() requires C<T>;
    // expected-note@-1 {{candidate function}}
};
template void a<int>::no_candidate();
// expected-error@-1 {{no viable candidate for explicit instantiation of 'no_candidate'}}

template void a<int>::ambiguous();
// expected-error@-1 {{partial ordering for explicit instantiation of 'ambiguous' is ambiguous}}

template void a<int>::ok();
template void a<int>::ok2();
