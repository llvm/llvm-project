// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:   -verify=other %s

void ok(void) __attribute__((hlx_vector_stack_limit(4096))); // other-warning {{unknown attribute 'hlx_vector_stack_limit' ignored}}

// expected-error@+2 {{'hlx_vector_stack_limit' attribute takes one argument}}
// other-warning@+1 {{unknown attribute 'hlx_vector_stack_limit' ignored}}
void no_arg(void) __attribute__((hlx_vector_stack_limit));

// expected-error@+2 {{'hlx_vector_stack_limit' attribute requires a non-negative integral compile time constant expression}}
// other-warning@+1 {{unknown attribute 'hlx_vector_stack_limit' ignored}}
void negative(void) __attribute__((hlx_vector_stack_limit(-1)));

// expected-warning@+2 {{'hlx_vector_stack_limit' attribute only applies to functions}}
// other-warning@+1 {{unknown attribute 'hlx_vector_stack_limit' ignored}}
int var __attribute__((hlx_vector_stack_limit(4096)));
