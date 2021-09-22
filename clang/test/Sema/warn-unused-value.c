// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -Wunused-value -Wunused-label %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -Wunused %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -Wall %s

int i = 0;
int j = 0;

void foo();

// PR4806
void pr4806() {
  1,foo();          // expected-warning {{left operand of comma operator has no effect}}

  // other
  foo();
  i;                // expected-warning {{expression result unused}}

  i,foo();          // expected-warning {{left operand of comma operator has no effect}}
  foo(),i;          // expected-warning {{expression result unused}}

  i,j,foo();        // expected-warning 2{{left operand of comma operator has no effect}}
  i,foo(),j;        // expected-warning {{left operand of comma operator has no effect}} expected-warning {{expression result unused}}
  foo(),i,j;        // expected-warning {{expression result unused}} expected-warning {{left operand of comma operator has no effect}}

  i++;

  i++,foo();
  foo(),i++;

  i++,j,foo();      // expected-warning {{left operand of comma operator has no effect}}
  i++,foo(),j;      // expected-warning {{expression result unused}}
  foo(),i++,j;      // expected-warning {{expression result unused}}

  i,j++,foo();      // expected-warning {{left operand of comma operator has no effect}}
  i,foo(),j++;      // expected-warning {{left operand of comma operator has no effect}}
  foo(),i,j++;      // expected-warning {{left operand of comma operator has no effect}}

  i++,j++,foo();
  i++,foo(),j++;
  foo(),i++,j++;

  {};
  ({});
  ({}),foo();
  foo(),({});

  (int)1U;          // expected-warning {{expression result unused}}
  (void)1U;

  // pointer to volatile has side effect (thus no warning)
  int* pi = &i;
  volatile int* pj = &j;
  *pi;              // expected-warning {{expression result unused}}
  *pj;

  foo_label:        // expected-warning {{unused label}}
  i;                // expected-warning {{expression result unused}}
}

// Don't warn about unused '||', '&&' expressions that contain assignments.
int test_logical_foo1();
int test_logical_foo2();
int test_logical_foo3();
int test_logical_bar() {
  int x = 0;
  (x = test_logical_foo1()) ||  // no-warning
  (x = test_logical_foo2()) ||  // no-warning
  (x = test_logical_foo3());    // no-warning

  x || test_logical_foo1();     // no-warning

  return x;
}

// PR8282
void conditional_for_control_flow(int cond, int x, int y)
{
    cond? y++ : x; // no-warning
    cond? y : ++x; // no-warning
    cond? (x |= y) : ++x; // no-warning
    cond? y : x; // expected-warning {{expression result unused}}
}

struct s0 { int f0; };

void f0(int a);
void f1(struct s0 *a) {
  // rdar://8139785
  f0((int)(a->f0 + 1, 10)); // expected-warning {{left operand of comma operator has no effect}}
}

void blah(int a);
#define GenTest(x) _Generic(x, default : blah)(x)

void unevaluated_operands(void) {
  int val = 0;

  (void)sizeof(++val); // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  (void)_Generic(val++, default : 0); // expected-warning {{expression with side effects has no effect in an unevaluated context}}
  (void)_Alignof(val++);  // expected-warning {{expression with side effects has no effect in an unevaluated context}} expected-warning {{'_Alignof' applied to an expression is a GNU extension}}

  // VLAs can have side effects so long as it's part of the type and not
  // an expression.
  (void)sizeof(int[++val]); // Ok
  (void)_Alignof(int[++val]); // Ok

  // Side effects as part of macro expansion are ok.
  GenTest(val++);
}
