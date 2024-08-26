// RUN: %clang_cc1 -std=c++2a -emit-llvm-only %s
// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fdelayed-template-parsing %s
// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fms-extensions %s
// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fdelayed-template-parsing -fms-extensions %s

namespace PR45881 {
struct A {
    void f();
};
int id(A*);
void A::f() {
    auto z = [*this](auto z2, decltype(z2(this)) z3){};
    z(id,3);
}

struct B {
    void f();
};
void B::f() {
    auto z = []<typename TT, typename TTT=decltype(TT()(this))>(){return 0;};
    z.template operator()<int(*)(B*)>();
}

struct C {
    void f();
};
void C::f() {
    auto z = []<typename TT, decltype(TT()(this)) n>(){return 0;};
    z.template operator()<int(*)(C*), 8>();
}
} // namespace PR45881
