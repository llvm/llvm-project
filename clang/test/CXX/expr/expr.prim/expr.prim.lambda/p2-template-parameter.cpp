// RUN: %clang_cc1 -std=c++17 %s -verify

template<auto> struct Nothing {};

void pr33696() {
    Nothing<[]() { return 0; }()> nothing; // expected-error{{a lambda expression cannot appear in this context}}
}
