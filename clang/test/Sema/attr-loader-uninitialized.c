// RUN: %clang_cc1 %s -verify -fsyntax-only
// See also attr-loader-uninitialized.cpp

int good __attribute__((loader_uninitialized));
static int local_ok __attribute__((loader_uninitialized));
int hidden_ok __attribute__((visibility("hidden"))) __attribute__((loader_uninitialized));

const int can_still_be_const __attribute__((loader_uninitialized));

extern int external_rejected __attribute__((loader_uninitialized));
// expected-error@-1 {{variable 'external_rejected' cannot be declared both 'extern' and with the 'loader_uninitialized' attribute}}

struct S;
extern struct S incomplete_external_rejected __attribute__((loader_uninitialized));
// expected-error@-1 {{variable 'incomplete_external_rejected' cannot be declared both 'extern' and with the 'loader_uninitialized' attribute}}

int noargs __attribute__((loader_uninitialized(0)));
// expected-error@-1 {{'loader_uninitialized' attribute takes no arguments}}

int init_rejected __attribute__((loader_uninitialized)) = 42;
// expected-error@-1 {{variable with 'loader_uninitialized' attribute cannot have an initializer}}

int declaration_then_uninit_ok;
int declaration_then_uninit_ok __attribute__((loader_uninitialized));

int definition_then_uninit_rejected = 0;
int definition_then_uninit_rejected __attribute__((loader_uninitialized));
// expected-error@-1 {{redeclaration cannot add 'loader_uninitialized' attribute}}
// expected-note@-3 {{previous definition is here}}

int tentative_repeated_ok __attribute__((loader_uninitialized));
int tentative_repeated_ok __attribute__((loader_uninitialized));

__private_extern__ int private_extern_can_be_initialised = 10;
__private_extern__ int therefore_uninit_private_extern_ok __attribute__((loader_uninitialized));

__private_extern__ int initialized_private_extern_rejected __attribute__((loader_uninitialized)) = 5;
// expected-error@-1 {{variable with 'loader_uninitialized' attribute cannot have an initializer}}

extern __attribute__((visibility("hidden"))) int extern_hidden __attribute__((loader_uninitialized));
// expected-error@-1 {{variable 'extern_hidden' cannot be declared both 'extern' and with the 'loader_uninitialized' attribute}}

struct Incomplete;
struct Incomplete incomplete __attribute__((loader_uninitialized));
// expected-error@-1 {{variable has incomplete type 'struct Incomplete'}}
// expected-note@-3 {{forward declaration of 'struct Incomplete'}}
