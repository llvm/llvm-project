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
    friend void g();

    temp(); // expected-note {{implicitly declared private here}}
};

template<C<temp<int>> T>
void g() {
    auto v = temp<T>(); // expected-error {{calling a private constructor of class 'temp<int>'}}
}

void h() {
    f<int>();
    g<int>(); // expected-note {{in instantiation of function template specialization 'g<int>' requested here}}
}
