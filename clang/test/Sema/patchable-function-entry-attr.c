// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify %s

// expected-error@+1 {{'patchable_function_entry' attribute takes at least 1 argument}}
__attribute__((patchable_function_entry)) void f();

// expected-error@+1 {{'patchable_function_entry' attribute takes no more than 2 arguments}}
__attribute__((patchable_function_entry(0, 0, 0))) void f();

// expected-error@+1 {{'patchable_function_entry' attribute requires a non-negative integral compile time constant expression}}
__attribute__((patchable_function_entry(-1))) void f();

int i;
// expected-error@+1 {{'patchable_function_entry' attribute requires parameter 0 to be an integer constant}}
__attribute__((patchable_function_entry(i))) void f();

// expected-error@+1 {{'patchable_function_entry' attribute requires integer constant between 0 and 2 inclusive}}
__attribute__((patchable_function_entry(2, 3))) void f();
