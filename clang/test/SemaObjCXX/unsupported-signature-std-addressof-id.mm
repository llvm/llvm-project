// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fobjc-runtime-has-weak -verify %s
// expected-no-diagnostics

namespace std {
    template <class T>
    T* addressof(T&);
}

void f(id obj) {
    (void)std::addressof(*obj);
}
