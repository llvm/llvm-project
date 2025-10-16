// RUN: %clang_cc1 %s -verify -fsyntax-only

[[clang::nooutline]] int a; // expected-error {{'clang::nooutline' attribute only applies to functions}}

[[clang::nooutline]] void t1(void);

[[clang::nooutline(2)]] void t2(void); // expected-error {{'clang::nooutline' attribute takes no arguments}}
