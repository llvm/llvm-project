// RUN: %clang_cc1 -verify -fsyntax-only %s -Wno-c++17-extensions

[[clang::nooutline]] int a; // expected-error {{'clang::nooutline' attribute only applies to functions}}

[[clang::nooutline]] void t1(void);

[[clang::nooutline(2)]] void t2(void); // expected-error {{'clang::nooutline' attribute takes no arguments}}
