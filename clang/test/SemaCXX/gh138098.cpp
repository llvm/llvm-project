// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template<typename T> struct Derived {
    void f(int index) {
        // expected-error@+3 {{explicit qualification required to use member 'type_c' from dependent base class}}
        // expected-error@+2 {{comparison between pointer and integer ('int' and 'auto &(*)()')}}
        // expected-note@8 {{in instantiation of member function 'Derived<int>::type_c' requested here}}
        if (index != type_c<T>) {}
    }
    // expected-error@+2 {{cannot deduce return type 'auto &' for function with no return statements}}
    // expected-note@12 {{member is declared here}}
    static auto& type_c() { }
};

int main() {
    Derived<int> d;
    // expected-note@19 {{in instantiation of member function 'Derived<int>::f' requested here}}
    // expected-note@19 {{in instantiation of member function 'Derived<int>::f' requested here}}
    d.f(0);
}
