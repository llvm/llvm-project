// RUN: %clang_cc1 -triple mips-linux-gnu  -fsyntax-only -verify %s

void __attribute__((target("arch=mips1")))
foo(void) {}
// expected-error@+3 {{function multiversioning is not supported on the current target}}
// expected-note@-2 {{previous declaration is here}}
void __attribute__((target("arch=mips2")))
foo(void) {}

// expected-error@+2 {{function multiversioning is not supported on the current target}}
void __attribute__((target("default")))
bar(void){}
