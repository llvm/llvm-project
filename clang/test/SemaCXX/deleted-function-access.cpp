// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

struct S {
    virtual void f() = delete; //expected-note{{'f' has been explicitly marked deleted here}}
    void g() { f(); } //expected-error{{attempt to use a deleted function}}
};
