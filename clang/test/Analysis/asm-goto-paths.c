// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify %s

void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached(void);

// The shape of a Linux kernel static branch.
static inline int static_branch(const int *key) {
  asm goto("jmp %l[l_yes]" : : "i"(key) : : l_yes);
  return 0;
l_yes:
  return 1;
}

int key;

void both_ways(void) {
  if (static_branch(&key))
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  clang_analyzer_warnIfReached();   // expected-warning {{REACHABLE}}
}

int null_after_asm_goto(int *p) {
  if (static_branch(&key))
    p = 0;
  return *p; // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
}

void outputs_are_forgotten(void) {
  int x = 1;

  asm goto("" : "=r"(x) : : : out);
  clang_analyzer_eval(x == 1); // expected-warning {{UNKNOWN}}
  return;
out:
  clang_analyzer_eval(x == 1); // expected-warning {{UNKNOWN}}
}

struct S { int a; };

void struct_outputs_are_forgotten(void) {
  struct S s = {1};

  asm goto("" : "=m"(s) : : : out);
  clang_analyzer_eval(s.a == 1); // expected-warning {{UNKNOWN}}
  return;
out:
  clang_analyzer_eval(s.a == 1); // expected-warning {{UNKNOWN}}
}

// Memory that an input points to may have been written, as for a plain asm.
void input_pointees_are_forgotten(int *p) {
  *p = 1;

  asm goto("" : : "r"(p) : "memory" : out);
  clang_analyzer_eval(*p == 1); // expected-warning {{UNKNOWN}}
  return;
out:
  clang_analyzer_eval(*p == 1); // expected-warning {{UNKNOWN}}
}

// The arms of a conditional operand are evaluated in earlier blocks. Only
// the memory behind the arm that was taken is forgotten.
void conditional_operand(int c, int *p, int *q) {
  *p = 1;
  *q = 1;

  asm goto("" : : "r"(c ? p : q) : "memory" : out);
  clang_analyzer_eval(*p == 1); // expected-warning {{TRUE}}
                                // expected-warning@-1 {{UNKNOWN}}
  clang_analyzer_eval(*q == 1); // expected-warning {{TRUE}}
                                // expected-warning@-1 {{UNKNOWN}}
  return;
out:
  clang_analyzer_eval(*p == 1); // expected-warning {{TRUE}}
                                // expected-warning@-1 {{UNKNOWN}}
  clang_analyzer_eval(*q == 1); // expected-warning {{TRUE}}
                                // expected-warning@-1 {{UNKNOWN}}
}

// A label before the asm goto makes a loop. The checks only run on the
// second time round: the backward edge is followed, the output gets a new
// value each time, and the path goes on behind the loop.
void backward_jump(void) {
  int x = 0;
  int prev;
  int count = 0;

again:
  prev = x;
  count++;
  asm goto("" : "=r"(x) : : : again);
  if (count == 2) {
    clang_analyzer_eval(prev == x); // expected-warning {{UNKNOWN}}
    clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  }
}

// No operands, so the block has no elements and there is nothing to
// invalidate. Both ways are still reachable.
void no_operands(void) {
  asm goto("jmp %l[out]" : : : : out);
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  return;
out:
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}
