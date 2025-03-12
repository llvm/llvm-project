// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class T, class V>
struct A{
    A();
    A(A&);
    A(A<V, T>); // expected-error{{copy constructor must pass its first argument by reference}}
};

void f() {
    A<int, int> a = A<int, int>(); // expected-note{{in instantiation of template class 'A<int, int>'}}
}

template<class T, class V>
struct B{
    B();
    template<class U> B(U); // No error (templated constructor)
};

void g() {
    B<int, int> b = B<int, int>(); // should use implicit copy constructor
}
