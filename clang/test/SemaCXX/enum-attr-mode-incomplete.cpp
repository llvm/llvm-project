// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify -std=c++17 \
// RUN:            -fmodules-local-submodule-visibility %s

// An attribute such as 'mode' gives a forward declared enum an underlying type,
// which makes the enum complete even though it never acquires a definition.
// Clang used to go looking for that non-existent definition and crash. The
// crash only happened with local submodule visibility enabled, which -std=c++20
// turns on implicitly.

typedef enum __attribute__((mode(TI))) MyEnum; // expected-error {{ISO C++ forbids forward references to 'enum' types}} \
                                               // expected-warning {{typedef requires a name}}
MyEnum x;
