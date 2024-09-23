// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-checker=core,alpha.security.ArrayBoundV2,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-checker=core,alpha.security.ArrayBoundV2,debug.ExprInspection -analyzer-config eagerly-assume=false -verify %s

// Note that eagerly-assume=false is tested separately because the
// WeakLoopAssumption suppression heuristic uses different code paths to
// achieve the same result with and without eagerly-assume.

void clang_analyzer_eval(int);

// Tests doing an out-of-bounds access after the end of an array using:
// - constant integer index
// - constant integer size for buffer
void test1(int x) {
  int buf[100];
  buf[100] = 1; // expected-warning{{Out of bound access to memory}}
}

void test1_ok(int x) {
  int buf[100];
  buf[99] = 1; // no-warning
}

const char test1_strings_underrun(int x) {
  const char *mystr = "mary had a little lamb";
  return mystr[-1]; // expected-warning{{Out of bound access to memory}}
}

const char test1_strings_overrun(int x) {
  const char *mystr = "mary had a little lamb";
  return mystr[1000];  // expected-warning{{Out of bound access to memory}}
}

const char test1_strings_ok(int x) {
  const char *mystr = "mary had a little lamb";
  return mystr[5]; // no-warning
}

// Tests doing an out-of-bounds access after the end of an array using:
// - indirect pointer to buffer
// - constant integer index
// - constant integer size for buffer
void test1_ptr(int x) {
  int buf[100];
  int *p = buf;
  p[101] = 1; // expected-warning{{Out of bound access to memory}}
}

void test1_ptr_ok(int x) {
  int buf[100];
  int *p = buf;
  p[99] = 1; // no-warning
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer, manipulated using simple pointer arithmetic
// - constant integer index
// - constant integer size for buffer
void test1_ptr_arith(int x) {
  int buf[100];
  int *p = buf;
  p = p + 100;
  p[0] = 1; // expected-warning{{Out of bound access to memory}}
}

void test1_ptr_arith_ok(int x) {
  int buf[100];
  int *p = buf;
  p = p + 99;
  p[0] = 1; // no-warning
}

void test1_ptr_arith_bad(int x) {
  int buf[100];
  int *p = buf;
  p = p + 99;
  p[1] = 1; // expected-warning{{Out of bound access to memory}}
}

void test1_ptr_arith_ok2(int x) {
  int buf[100];
  int *p = buf;
  p = p + 99;
  p[-1] = 1; // no-warning
}

// Tests doing an out-of-bounds access before the start of an array using:
// - constant integer index
// - constant integer size for buffer
void test2(int x) {
  int buf[100];
  buf[-1] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer
// - constant integer index
// - constant integer size for buffer
void test2_ptr(int x) {
  int buf[100];
  int *p = buf;
  p[-1] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests doing an out-of-bounds access before the start of an array using:
// - indirect pointer to buffer, manipulated using simple pointer arithmetic
// - constant integer index
// - constant integer size for buffer
void test2_ptr_arith(int x) {
  int buf[100];
  int *p = buf;
  --p;
  p[0] = 1; // expected-warning {{Out of bound access to memory preceding}}
}

// Tests doing an out-of-bounds access before the start of a multi-dimensional
// array using:
// - constant integer indices
// - constant integer sizes for the array
void test2_multi(int x) {
  int buf[100][100];
  buf[0][-1] = 1; // expected-warning{{Out of bound access to memory}}
}

// Tests doing an out-of-bounds access before the start of a multi-dimensional
// array using:
// - constant integer indices
// - constant integer sizes for the array
void test2_multi_b(int x) {
  int buf[100][100];
  buf[-1][0] = 1; // expected-warning{{Out of bound access to memory}}
}

void test2_multi_ok(int x) {
  int buf[100][100];
  buf[0][0] = 1; // no-warning
}

void test3(int x) {
  int buf[100];
  if (x < 0)
    buf[x] = 1; // expected-warning{{Out of bound access to memory}}
}

void test4(int x) {
  int buf[100];
  if (x > 99)
    buf[x] = 1; // expected-warning{{Out of bound access to memory}}
}

void test_assume_after_access(unsigned long x) {
  int buf[100];
  buf[x] = 1;
  clang_analyzer_eval(x <= 99); // expected-warning{{TRUE}}
}

// Don't warn when indexing below the start of a symbolic region's whose
// base extent we don't know.
int *get_symbolic(void);
void test_underflow_symbolic(void) {
  int *buf = get_symbolic();
  buf[-1] = 0; // no-warning;
}

// But warn if we understand the internal memory layout of a symbolic region.
typedef struct {
  int id;
  char name[256];
} user_t;

user_t *get_symbolic_user(void);
char test_underflow_symbolic_2() {
  user_t *user = get_symbolic_user();
  return user->name[-1]; // expected-warning{{Out of bound access to memory}}
}

void test_incomplete_struct(void) {
  extern struct incomplete incomplete;
  int *p = (int *)&incomplete;
  p[1] = 42; // no-warning
}

void test_extern_void(void) {
  extern void v;
  int *p = (int *)&v;
  p[1] = 42; // no-warning
}

void test_assume_after_access2(unsigned long x) {
  char buf[100];
  buf[x] = 1;
  clang_analyzer_eval(x <= 99); // expected-warning{{TRUE}}
}

struct incomplete;
char test_comparison_with_extent_symbol(struct incomplete *p) {
  // Previously this was reported as a (false positive) overflow error because
  // the extent symbol of the area pointed by `p` was an unsigned and the '-1'
  // was converted to its type by `evalBinOpNN`.
  return ((char *)p)[-1]; // no-warning
}

// WeakLoopAssumption suppression
///////////////////////////////////////////////////////////////////////

int GlobalArray[100];
int loop_suppress_after_zero_iterations(unsigned len) {
  for (unsigned i = 0; i < len; i++)
    if (GlobalArray[i] > 0)
      return GlobalArray[i];
  // Previously this would have produced an overflow warning because splitting
  // the state on the loop condition introduced an execution path where the
  // analyzer thinks that len == 0.
  // There are very many situations where the programmer knows that an argument
  // is positive, but this is not indicated in the source code, so we must
  // avoid reporting errors (especially out of bounds errors) on these
  // branches, because otherwise we'd get prohibitively many false positives.
  return GlobalArray[len - 1]; // no-warning
}

void loop_report_in_second_iteration(int len) {
  int buf[1] = {0};
  for (int i = 0; i < len; i++) {
    // When a programmer writes a loop, we may assume that they intended at
    // least two iterations.
    buf[i] = 1; // expected-warning{{Out of bound access to memory}}
  }
}

void loop_suppress_in_third_iteration(int len) {
  int buf[2] = {0};
  for (int i = 0; i < len; i++) {
    // We should suppress array bounds errors on the third and later iterations
    // of loops, because sometimes programmers write a loop in sitiuations
    // where they know that there will be at most two iterations.
    buf[i] = 1; // no-warning
  }
}

void loop_suppress_in_third_iteration_cast(int len) {
  int buf[2] = {0};
  for (int i = 0; (unsigned)(i < len); i++) {
    // Check that a (somewhat arbitrary) cast does not hinder the recognition
    // of the condition expression.
    buf[i] = 1; // no-warning
  }
}

void loop_suppress_in_third_iteration_logical_and(int len, int flag) {
  int buf[2] = {0};
  for (int i = 0; i < len && flag; i++) {
    // FIXME: In this case the checker should suppress the warning the same way
    // as it's suppressed in loop_suppress_in_third_iteration, but the
    // suppression is not activated because the terminator statement associated
    // with the loop is just the expression 'flag', while 'i < len' is a
    // separate terminator statement that's associated with the
    // short-circuiting operator '&&'.
    // I have seen a real-world FP that looks like this, but it is much rarer
    // than the basic setup.
    buf[i] = 1; // expected-warning{{Out of bound access to memory}}
  }
}

void loop_suppress_in_third_iteration_logical_and_2(int len, int flag) {
  int buf[2] = {0};
  for (int i = 0; flag && i < len; i++) {
    // If the two operands of '&&' are flipped, the suppression works.
    buf[i] = 1; // no-warning
  }
}

int coinflip(void);
int do_while_report_after_one_iteration(void) {
  int i = 0;
  do {
    i++;
  } while (coinflip());
  // Unlike `loop_suppress_after_zero_iterations`, running just one iteration
  // in a do-while is not a corner case that would produce too many false
  // positives, so don't suppress bounds errors in these situations.
  return GlobalArray[i-2]; // expected-warning{{Out of bound access to memory}}
}

void do_while_report_in_second_iteration(int len) {
  int buf[1] = {0};
  int i = 0;
  do {
    buf[i] = 1; // expected-warning{{Out of bound access to memory}}
  } while (i++ < len);
}

void do_while_suppress_in_third_iteration(int len) {
  int buf[2] = {0};
  int i = 0;
  do {
    buf[i] = 1; // no-warning
  } while (i++ < len);
}
