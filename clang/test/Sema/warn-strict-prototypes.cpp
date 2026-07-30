// RUN: %clang_cc1 -verify -fsyntax-only -Wstrict-prototypes %s
// expected-no-diagnostics

void decl();
void decl_void(void);

void def() {}
void def_void(void) {}
