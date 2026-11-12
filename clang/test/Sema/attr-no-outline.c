// RUN: %clang_cc1 -verify -fsyntax-only %s

[[clang::no_outline]] int a; // expected-error {{'clang::no_outline' attribute only applies to functions}}

[[clang::no_outline]] void t1(void);

[[clang::no_outline(2)]] void t2(void); // expected-error {{'clang::no_outline' attribute takes no arguments}}
