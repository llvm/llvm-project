// RUN: %clang_cc1 -std=c++20 -verify %s

template<class T, class U>
concept C = true;

class non_temp {
    template<C<non_temp> T>
    friend void f();

    non_temp();
};

template<C<non_temp> T>
void f() {
    auto v = non_temp();
}

template<class A>
class temp {
    template<C<temp> T>
    friend void g(); // expected-error {{friend declaration with a constraint that depends on an enclosing template parameter must be a definition}}

    temp();
};

template<C<temp<int>> T>
void g() {
    auto v = temp<T>();
}

void h() {
    f<int>();
    g<int>();
}
