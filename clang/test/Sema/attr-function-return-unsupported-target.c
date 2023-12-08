// RUN: %clang_cc1 -triple s390x-linux-gnu -fsyntax-only -verify %s

// expected-warning@+1 {{unknown attribute 'function_return' ignored}}
__attribute__((function_return("keep"))) void x(void) {}

// expected-warning@+1 {{unknown attribute 'function_return' ignored}}
__attribute__((function_return("thunk"))) void y(void) {}

// expected-warning@+1 {{unknown attribute 'function_return' ignored}}
__attribute__((function_return("thunk-inline"))) void z(void) {}

// expected-warning@+1 {{unknown attribute 'function_return' ignored}}
__attribute__((function_return("thunk-extern"))) void w(void) {}

// expected-warning@+1 {{unknown attribute 'function_return' ignored}}
__attribute__((function_return("invalid"))) void v(void) {}
