// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/usings.cppm -o %t/usings.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=usings=%t/usings.pcm %t/use.cpp -verify -fsyntax-only -Wno-stack-exhausted

// expected-no-diagnostics

//--- usings.cppm
export module usings;

#define TYPES1(NAME) DECLARE(NAME##a) DECLARE(NAME##b) DECLARE(NAME##c) \
    DECLARE(NAME##d) DECLARE(NAME##e) DECLARE(NAME##f) DECLARE(NAME##g) \
    DECLARE(NAME##h) DECLARE(NAME##i) DECLARE(NAME##j) 
#define TYPES2(NAME) TYPES1(NAME##a) TYPES1(NAME##b) TYPES1(NAME##c) \
    TYPES1(NAME##d) TYPES1(NAME##e) TYPES1(NAME##f) TYPES1(NAME##g) \
    TYPES1(NAME##h) TYPES1(NAME##i) TYPES1(NAME##j) 
#define TYPES3(NAME) TYPES2(NAME##a) TYPES2(NAME##b) TYPES2(NAME##c) \
    TYPES2(NAME##d) TYPES2(NAME##e) TYPES2(NAME##f) TYPES2(NAME##g) \
    TYPES2(NAME##h) TYPES2(NAME##i) TYPES2(NAME##j) 
#define TYPES4(NAME) TYPES3(NAME##a) TYPES3(NAME##b) TYPES3(NAME##c) \
    TYPES3(NAME##d) TYPES3(NAME##e) TYPES3(NAME##f) TYPES3(NAME##g)

#define DECLARE(NAME) struct NAME {};
TYPES4(Type)

export struct Base {
#undef DECLARE
#define DECLARE(NAME) void func(NAME*);
TYPES4(Type)
};

export struct Derived : Base {
    using Base::func;
};

//--- use.cpp
import usings;
void test() {
    Derived().func(nullptr); // expected-error{{ambiguous}}
    // expected-note@* + {{candidate function}}
}
