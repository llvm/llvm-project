// RUN: %clang_cc1 -fsyntax-only -verify %s

void __attribute__((no_stack_protector)) foo(void) {}
int __attribute__((no_stack_protector)) var; // expected-warning {{'no_stack_protector' attribute only applies to functions}}
void  __attribute__((no_stack_protector(2))) bar(void) {} // expected-error {{'no_stack_protector' attribute takes no arguments}}
