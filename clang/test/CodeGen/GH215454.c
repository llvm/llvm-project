// RUN: %clang_cc1 -std=gnu99 -verify -emit-llvm-only %s
// RUN: %clang_cc1 -std=gnu99 -DCODEGEN -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// A declaration that declares nothing as the last statement of a statement
// expression made the whole statement expression invalid without an error,
// which dropped the call below or crashed CodeGen on a RecoveryExpr.

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
