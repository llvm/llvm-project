// RUN: %clang_cc1 -std=c++2c -verify=cxx26 -fsyntax-only -Wpre-c++26-compat %s
// RUN: %clang_cc1 -std=c++11 -verify=cxx11 -fsyntax-only -Wc++26-extensions %s

template <typename... T>
void f(T... t) {
    // cxx26-warning@+2 {{pack indexing is incompatible with C++ standards before C++2c}}
    // cxx11-warning@+1 {{pack indexing is a C++2c extension}}
    using a = T...[0];

    // cxx26-warning@+2 {{pack indexing is incompatible with C++ standards before C++2c}}
    // cxx11-warning@+1 {{pack indexing is a C++2c extension}}
    using b = typename T...[0]::a;

    // cxx26-warning@+2 2{{pack indexing is incompatible with C++ standards before C++2c}}
    // cxx11-warning@+1 2{{pack indexing is a C++2c extension}}
    t...[0].~T...[0]();

    // cxx26-warning@+2 {{pack indexing is incompatible with C++ standards before C++2c}}
    // cxx11-warning@+1 {{pack indexing is a C++2c extension}}
    T...[0] c;
}
