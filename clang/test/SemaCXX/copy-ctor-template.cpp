// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class T, class V>
struct A{
    A(); // expected-note{{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
    A(A&); // expected-note{{candidate constructor not viable: expects an lvalue for 1st argument}} 
    A(A<V, T>); // expected-error{{copy constructor must pass its first argument by reference}}
};

void f() {
    A<int, int> a = A<int, int>(); // expected-note{{in instantiation of template class 'A<int, int>'}} 
    A<int, double> a1 = A<double, int>(); // No error (not a copy constructor)
}

// Test rvalue-to-lvalue conversion in copy constructor
A<int, int> &&a(void);
void g() {
    A<int, int> a2 = a(); // expected-error{{no matching constructor}}
}

template<class T, class V>
struct B{
    B();
    template<class U> B(U); // No error (templated constructor)
};

void h() {
    B<int, int> b = B<int, int>(); // should use implicit copy constructor
}
