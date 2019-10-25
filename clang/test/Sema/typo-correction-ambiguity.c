// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check the following typo correction behavior in C:
// - no typos are diagnosed when a call expression has ambiguous (multiple) corrections

int v_63;

void v_2_0(int v_452, int v_454) {}

int v_3_0() {
   for (int v_345 = 0 ; v_63;)
       v_2_0(v_195,  // expected-error {{use of undeclared identifier 'v_195'}}
             v_231);  // expected-error {{use of undeclared identifier 'v_231'}}
}
