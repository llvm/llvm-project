// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify %s

__attribute__((function_return("keep"))) void x(void) {}

// expected-warning@+1 {{'function_return' attribute argument not supported: thunk}}
__attribute__((function_return("thunk"))) void y(void) {}

// expected-warning@+1 {{'function_return' attribute argument not supported: thunk-inline}}
__attribute__((function_return("thunk-inline"))) void z(void) {}

__attribute__((function_return("thunk-extern"))) void w(void) {}

// expected-warning@+1 {{'function_return' attribute argument not supported: invalid}}
__attribute__((function_return("invalid"))) void v(void) {}

// expected-error@+1 {{'function_return' attribute requires a string}}
__attribute__((function_return(5))) void a(void) {}

// expected-error@+1 {{'function_return' attribute takes one argument}}
__attribute__((function_return)) void b(void) {}

// expected-warning@+1 {{'function_return' attribute only applies to functions}}
__attribute__((function_return)) int c;
