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

// Test: no typo-correction diagnostics are emitted for ambiguous typos.
struct a {
  int xxx;
};

int g_107;
int g_108;
int g_109;

struct a g_999;
struct a g_998;
void PR50797() { (g_910.xxx = g_910.xxx); } //expected-error 2{{use of undeclared identifier 'g_910'}}
