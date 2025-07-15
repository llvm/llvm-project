// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -fblocks -fcxx-exceptions -fms-extensions -verify -Wfunction-effects %s

#pragma clang diagnostic ignored "-Wperf-constraint-implies-noexcept"

// These need '-fms-extensions' (and maybe '-fdeclspec')
void f1() [[clang::nonblocking]] {
    __try {} __except (1) {} // expected-warning {{function with 'nonblocking' attribute must not throw or catch exceptions}}
}

struct S {
    int get_x(); // expected-note {{declaration cannot be inferred 'nonblocking' because it has no definition in this translation unit}}
    __declspec(property(get = get_x)) int x;

    int get_nb() { return 42; }
    __declspec(property(get = get_nb)) int nb;

    int get_nb2() [[clang::nonblocking]];
    __declspec(property(get = get_nb2)) int nb2;
};

void f2() [[clang::nonblocking]] {
    S a;
    a.x; // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function 'S::get_x'}}
    a.nb;
    a.nb2;
}
