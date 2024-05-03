// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++23 -verify %s
// expected-no-diagnostics

struct B {
    template <typename S>
    void foo();

    void bar();
};

template <typename T, typename S>
struct A : T {
    auto foo() {
        static_assert(requires { T::template foo<S>(); });
        static_assert(requires { T::bar(); });
    }
};

int main() {
    A<B, double> a;
    a.foo();
}
