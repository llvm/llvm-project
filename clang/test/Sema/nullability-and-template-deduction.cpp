// RUN: %clang_cc1 -fsyntax-only %s -verify
// expected-no-diagnostics

template <class T> struct Base {};
template <class T> struct Derived : Base<T> {};

template <class T> void foo(Base<T> *_Nonnull);

template <class T> void bar(Base<T> *);


void test() {
    Derived<int> d;
    foo(&d);
    bar(&d);
}
