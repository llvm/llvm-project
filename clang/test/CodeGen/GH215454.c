// RUN: %clang_cc1 -std=gnu99 -verify -emit-llvm-only %s
// RUN: %clang_cc1 -std=gnu99 -verify -emit-llvm-only -fno-recovery-ast %s
// RUN: %clang_cc1 -std=gnu99 -DCODEGEN -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// A declaration that declares nothing made the enclosing statement expression
// invalid without an error: the call below was dropped, and as an 'if'
// condition CodeGen crashed on a RecoveryExpr.

void foo(void);

// CHECK-LABEL: define{{.*}} void @keeps_side_effects(
// CHECK: call void @foo()
void keeps_side_effects(int e) {
  ({ foo(); __typeof__(e); }); // expected-warning {{declaration does not declare anything}}
}

#ifndef CODEGEN
void d2(int e) {
  if (({ ; __typeof__(e); })) {} // expected-warning {{declaration does not declare anything}} \
                                 // expected-error {{statement requires expression of scalar type ('void' invalid)}}
}

// Reproducer from the issue; the unclosed '({' makes recovery run to EOF.
#define c(a, b)                                                                \
  {;__typeof__(b);}
void d(int e) {if((c(, e););  // expected-warning {{'(' and '{' tokens introducing statement expression appear in different macro expansion contexts}} \
                              // expected-note {{'{' token is here}} \
                              // expected-warning {{declaration does not declare anything}} \
                              // expected-error {{unexpected ';' before ')'}} \
                              // expected-note {{to match this '{'}}
}                             // expected-error {{expected expression}} \
                              // expected-error@+2 {{expected '}'}}
#endif
