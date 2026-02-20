// RUN: %clang_cc1 -fsyntax-only -Wpointer-bool-conversion %s
bool f() {
    int *p;
    if (p) {} // expected-warning {{implicit conversion of pointer to bool}}
    return (void *)0; // expected-warning {{implicit conversion of pointer to bool}}
}

bool g() {
    int *p = nullptr;
    if (p == nullptr) {} // no-warning
    return (void *)0 == nullptr; // no-warning
}
