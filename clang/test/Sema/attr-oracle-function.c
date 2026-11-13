// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -verify -fsyntax-only %s

[[clang::oracle_function]] long a; // expected-error {{'clang::oracle_function' attribute only applies to functions}}

[[clang::oracle_function]] long t1(void) {};// expected-error {{'clang::oracle_function' on a function with external linkage is not allowed}}

[[clang::oracle_function(2)]] long t2(void) {} // expected-error {{'clang::oracle_function' attribute takes no arguments}}

[[clang::oracle_function]] static long t3() {} // expected-error {{'clang::oracle_function' on a function with no prototype is not allowed}}

[[clang::oracle_function]] static long t4(int a, ...) {} // expected-error {{'clang::oracle_function' on a variadic function is not allowed}}

[[clang::oracle_function]] static void* t5(void) {} //expected-error {{'clang::oracle_function' can only be put on functions that return a 64-bit integer}}

[[clang::oracle_function]] static long t6(void) {}
