// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s

int foo([[maybe_unused, maybe_unused]] int a) { // expected-warning {{attribute 'maybe_unused' is already applied}}
    return 1;
}

[[noreturn, noreturn]] void g() { // expected-warning {{attribute 'noreturn' is already applied}}
    __builtin_unreachable();
}

int h(int n) {
    switch (n) {
    case 1:
        [[fallthrough, fallthrough]]; // expected-warning {{attribute 'fallthrough' is already applied}}
    case 2:
        return 1;
    }
    return 0;
}