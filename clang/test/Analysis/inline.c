// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);
void clang_analyzer_checkInlined(int);

int test1_f1() {
  int y = 1;
  y++;
  clang_analyzer_checkInlined(1); // expected-warning{{TRUE}}
  return y;
}

void test1_f2() {
  int x = 1;
  x = test1_f1();
  if (x == 1) {
    int *p = 0;
    *p = 3; // no-warning
  }
  if (x == 2) {
    int *p = 0;
    *p = 3; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
  }
}

// Test that inlining works when the declared function has less arguments
// than the actual number in the declaration.
void test2_f1() {}
int test2_f2();

void test2_f3() { 
  test2_f1(test2_f2()); // expected-warning{{too many arguments in call to 'test2_f1'}}
}

// Test that inlining works with recursive functions.

unsigned factorial(unsigned x) {
  if (x <= 1)
    return 1;
  return x * factorial(x - 1);
}

void test_factorial() {
  if (factorial(3) == 6) {
    int *p = 0;
    *p = 0xDEADBEEF;  // expected-warning {{null}}
  }
  else {
    int *p = 0;
    *p = 0xDEADBEEF; // no-warning
  }
}

void test_factorial_2() {
  unsigned x = factorial(3);
  if (x == factorial(3)) {
    int *p = 0;
    *p = 0xDEADBEEF;  // expected-warning {{null}}
  }
  else {
    int *p = 0;
    *p = 0xDEADBEEF; // no-warning
  }
}

// Test that returning stack memory from a parent stack frame does
// not trigger a warning.
static char *return_buf(char *buf) {
  return buf + 10;
}

void test_return_stack_memory_ok() {
  char stack_buf[100];
  char *pos = return_buf(stack_buf);
  (void) pos;
}

char *test_return_stack_memory_bad() {
  char stack_buf[100];
  char *x = stack_buf;
  return x; // expected-warning {{stack memory associated}}
}

// Test that passing a struct value with an uninitialized field does
// not trigger a warning if we are inlining and the body is available.
struct rdar10977037 { int x, y; };
int test_rdar10977037_aux(struct rdar10977037 v) { return v.y; }
int test_rdar10977037_aux_2(struct rdar10977037 v);
int test_rdar10977037() {
  struct rdar10977037 v;
  v.y = 1;
  v. y += test_rdar10977037_aux(v); // no-warning
  return test_rdar10977037_aux_2(v); // expected-warning {{Passed-by-value struct argument contains uninitialized data}}
}


// Test inlining a forward-declared function.
// This regressed when CallEvent was first introduced.
int plus1(int x);
void test() {
  clang_analyzer_eval(plus1(2) == 3); // expected-warning{{TRUE}}
}

int plus1(int x) {
  return x + 1;
}


void never_called_by_anyone() {
  clang_analyzer_checkInlined(0); // no-warning
}


void knr_one_argument(a) int a; { }

void call_with_less_arguments() {
  knr_one_argument(); // expected-warning{{too few arguments}} expected-warning{{Function taking 1 argument is called with fewer (0)}}
}
