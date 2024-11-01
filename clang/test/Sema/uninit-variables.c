// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -Wconditional-uninitialized -fsyntax-only -fblocks %s -verify
// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -Wconditional-uninitialized -ftrivial-auto-var-init=pattern -fsyntax-only -fblocks %s -verify

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

int test1(void) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  return x; // expected-warning{{variable 'x' is uninitialized when used here}}
}

int test2(void) {
  int x = 0;
  return x; // no-warning
}

int test3(void) {
  int x;
  x = 0;
  return x; // no-warning
}

int test4(void) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  ++x; // expected-warning{{variable 'x' is uninitialized when used here}}
  return x; 
}

int test5(void) {
  int x, y; // expected-note{{initialize the variable 'y' to silence this warning}}
  x = y; // expected-warning{{variable 'y' is uninitialized when used here}}
  return x;
}

int test6(void) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  x += 2; // expected-warning{{variable 'x' is uninitialized when used here}}
  return x;
}

int test7(int y) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  if (y) // expected-warning{{variable 'x' is used uninitialized whenever 'if' condition is false}} \
         // expected-note{{remove the 'if' if its condition is always true}}
    x = 1;
  return x; // expected-note{{uninitialized use occurs here}}
}

int test7b(int y) {
  int x = x; // expected-note{{variable 'x' is declared here}}
  if (y)
    x = 1;
  // Warn with "may be uninitialized" here (not "is sometimes uninitialized"),
  // since the self-initialization is intended to suppress a -Wuninitialized
  // warning.
  return x; // expected-warning{{variable 'x' may be uninitialized when used here}}
}

int test8(int y) {
  int x;
  if (y)
    x = 1;
  else
    x = 0;
  return x;
}

int test9(int n) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  for (unsigned i = 0 ; i < n; ++i) {
    if (i == n - 1)
      break;
    x = 1;
  }
  return x; // expected-warning{{variable 'x' may be uninitialized when used here}}
}

int test10(unsigned n) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  for (unsigned i = 0 ; i < n; ++i) {
    x = 1;
  }
  return x; // expected-warning{{variable 'x' may be uninitialized when used here}}
}

int test11(unsigned n) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  for (unsigned i = 0 ; i <= n; ++i) {
    x = 1;
  }
  return x; // expected-warning{{variable 'x' may be uninitialized when used here}}
}

void test12(unsigned n) {
  for (unsigned i ; n ; ++i) ; // expected-warning{{variable 'i' is uninitialized when used here}} expected-note{{initialize the variable 'i' to silence this warning}}
}

int test13(void) {
  static int i;
  return i; // no-warning
}

// Simply don't crash on this test case.
void test14(void) {
  const char *p = 0;
  for (;;) {}
}

void test15(void) {
  int x = x; // no-warning: signals intended lack of initialization.
}

int test15b(void) {
  // Warn here with the self-init, since it does result in a use of
  // an uninitialized variable and this is the root cause.
  int x = x; // expected-warning {{variable 'x' is uninitialized when used within its own initialization}}
  return x;
}

// Don't warn in the following example; shows dataflow confluence.
char *test16_aux(void);
void test16(void) {
  char *p = test16_aux();
  for (unsigned i = 0 ; i < 100 ; i++)
    p[i] = 'a'; // no-warning
}

void test17(void) {
  // Don't warn multiple times about the same uninitialized variable
  // along the same path.
  int *x; // expected-note{{initialize the variable 'x' to silence this warning}}
  *x = 1; // expected-warning{{variable 'x' is uninitialized when used here}}
  *x = 1; // no-warning
}

int test18(int x, int y) {
  int z;
  if (x && y && (z = 1)) {
    return z; // no-warning
  }
  return 0;
}

int test19_aux1(void);
int test19_aux2(void);
int test19_aux3(int *x);
int test19(void) {
  int z;
  if (test19_aux1() + test19_aux2() && test19_aux1() && test19_aux3(&z))
    return z; // no-warning
  return 0;
}

int test20(void) {
  int z; // expected-note{{initialize the variable 'z' to silence this warning}}
  if ((test19_aux1() + test19_aux2() && test19_aux1()) || test19_aux3(&z)) // expected-warning {{variable 'z' is used uninitialized whenever '||' condition is true}} expected-note {{remove the '||' if its condition is always false}}
    return z; // expected-note {{uninitialized use occurs here}}
  return 0;
}

int test21(int x, int y) {
  int z; // expected-note{{initialize the variable 'z' to silence this warning}}
  if ((x && y) || test19_aux3(&z) || test19_aux2()) // expected-warning {{variable 'z' is used uninitialized whenever '||' condition is true}} expected-note {{remove the '||' if its condition is always false}}
    return z; // expected-note {{uninitialized use occurs here}}
  return 0;
}

int test22(void) {
  int z;
  while (test19_aux1() + test19_aux2() && test19_aux1() && test19_aux3(&z))
    return z; // no-warning
  return 0;
}

int test23(void) {
  int z;
  for ( ; test19_aux1() + test19_aux2() && test19_aux1() && test19_aux3(&z) ; )
    return z; // no-warning
  return 0;
}

// The basic uninitialized value analysis doesn't have enough path-sensitivity
// to catch initializations relying on control-dependencies spanning multiple
// conditionals.  This possibly can be handled by making the CFG itself
// represent such control-dependencies, but it is a niche case.
int test24(int flag) {
  unsigned val; // expected-note{{initialize the variable 'val' to silence this warning}}
  if (flag)
    val = 1;
  if (!flag)
    val = 1;
  return val; // expected-warning{{variable 'val' may be uninitialized when used here}}
}

float test25(void) {
  float x; // expected-note{{initialize the variable 'x' to silence this warning}}
  return x; // expected-warning{{variable 'x' is uninitialized when used here}}
}

typedef int MyInt;
MyInt test26(void) {
  MyInt x; // expected-note{{initialize the variable 'x' to silence this warning}}
  return x; // expected-warning{{variable 'x' is uninitialized when used here}}
}

// Test handling of sizeof().
int test27(void) {
  struct test_27 { int x; } *y;
  return sizeof(y->x); // no-warning
}

int test28(void) {
  int len; // expected-note{{initialize the variable 'len' to silence this warning}}
  return sizeof(int[len]); // expected-warning{{variable 'len' is uninitialized when used here}}
}

void test29(void) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  (void) ^{ (void) x; }; // expected-warning{{variable 'x' is uninitialized when captured by block}}
}

void test30(void) {
  static int x; // no-warning
  (void) ^{ (void) x; };
}

void test31(void) {
  __block int x; // no-warning
  (void) ^{ (void) x; };
}

int test32_x;
void test32(void) {
  (void) ^{ (void) test32_x; }; // no-warning
}

void test_33(void) {
  int x; // no-warning
  (void) x;
}

int test_34(void) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  (void) x;
  return x; // expected-warning{{variable 'x' is uninitialized when used here}}
}

// Test that this case doesn't crash.
void test35(int x) {
  __block int y = 0;
  ^{ y = (x == 0); }();
}

// Test handling of indirect goto.
void test36(void)
{
  void **pc; // expected-note{{initialize the variable 'pc' to silence this warning}}
  void *dummy[] = { &&L1, &&L2 };
 L1:
    goto *pc; // expected-warning{{variable 'pc' is uninitialized when used here}}
 L2:
    goto *pc;
}

// Test && nested in ||.
int test37_a(void);
int test37_b(void);
int test37(void)
{
    int identifier;
    if ((test37_a() && (identifier = 1)) ||
        (test37_b() && (identifier = 2))) {
        return identifier; // no-warning
    }
    return 0;
}

// Test merging of path-specific dataflow values (without asserting).
int test38(int r, int x, int y)
{
  int z;
  return ((r < 0) || ((r == 0) && (x < y)));
}

int test39(int x) {
  int y; // expected-note{{initialize the variable 'y' to silence this warning}}
  int z = x + y; // expected-warning {{variable 'y' is uninitialized when used here}}
  return z;
}


int test40(int x) {
  int y; // expected-note{{initialize the variable 'y' to silence this warning}}
  return x ? 1 : y; // expected-warning {{variable 'y' is uninitialized when used here}}
}

int test41(int x) {
  int y; // expected-note{{initialize the variable 'y' to silence this warning}}
  if (x) y = 1; // expected-warning{{variable 'y' is used uninitialized whenever 'if' condition is false}} \
                // expected-note{{remove the 'if' if its condition is always true}}
  return y; // expected-note{{uninitialized use occurs here}}
}

void test42(void) {
  int a;
  a = 30; // no-warning
}

void test43_aux(int x);
void test43(int i) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  for (i = 0 ; i < 10; i++)
    test43_aux(x++); // expected-warning {{variable 'x' is uninitialized when used here}}
}

void test44(int i) {
  int x = i;
  int y; // expected-note{{initialize the variable 'y' to silence this warning}}
  for (i = 0; i < 10; i++ ) {
    test43_aux(x++); // no-warning
    x += y; // expected-warning {{variable 'y' is uninitialized when used here}}
  }
}

int test45(int j) {
  int x = 1, y = x + 1;
  if (y) // no-warning
    return x;
  return y;
}

void test46(void)
{
  int i; // expected-note{{initialize the variable 'i' to silence this warning}}
  int j = i ? : 1; // expected-warning {{variable 'i' is uninitialized when used here}}
}

void *test47(int *i)
{
  return i ? : 0; // no-warning
}

void *test49(int *i)
{
  int a;
  return &a ? : i; // no-warning
}

void test50(void)
{
  char c[1 ? : 2]; // no-warning
}

int test51(void)
{
    __block int a;
    ^(void) {
      a = 42;
    }();
    return a; // no-warning
}

// FIXME: This is a false positive, but it tests logical operations in switch statements.
int test52(int a, int b) {
  int x;  // expected-note {{initialize the variable 'x' to silence this warning}}
  switch (a || b) { // expected-warning {{switch condition has boolean value}}
    case 0:
      x = 1;
      break;
    case 1:
      x = 2;
      break;
  }
  return x; // expected-warning {{variable 'x' may be uninitialized when used here}}
}

void test53(void) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
  int y = (x);  // expected-warning {{variable 'x' is uninitialized when used here}}
}

// This CFG caused the uninitialized values warning to inf-loop.
extern int PR10379_g(void);
void PR10379_f(int *len) {
  int new_len; // expected-note{{initialize the variable 'new_len' to silence this warning}}
  for (int i = 0; i < 42 && PR10379_g() == 0; i++) {
    if (PR10379_g() == 1)
      continue;
    if (PR10379_g() == 2)
      PR10379_f(&new_len);
    else if (PR10379_g() == 3)
      PR10379_f(&new_len);
    *len += new_len; // expected-warning {{variable 'new_len' may be uninitialized when used here}}
  }
}

// Test that sizeof(VLA) doesn't trigger a warning.
void test_vla_sizeof(int x) {
  double (*memory)[2][x] = malloc(sizeof(*memory)); // no-warning
}

// Test absurd case of deadcode + use of blocks.  This previously was a false positive
// due to an analysis bug.
int test_block_and_dead_code(void) {
  __block int x;
  ^{ x = 1; }();
  if (0)
    return x;
  return x; // no-warning
}

// This previously triggered an infinite loop in the analysis.
void PR11069(int a, int b) {
  unsigned long flags;
  for (;;) {
    if (a && !b)
      break;
  }
  for (;;) {
    // This does not trigger a warning because it isn't a real use.
    (void)(flags); // no-warning
  }
}

// Test uninitialized value used in loop condition.
void rdar9432305(float *P) {
  int i; // expected-note {{initialize the variable 'i' to silence this warning}}
  for (; i < 10000; ++i) // expected-warning {{variable 'i' is uninitialized when used here}}
    P[i] = 0.0f;
}

// Test that fixits are not emitted inside macros.
#define UNINIT(T, x, y) T x; T y = x;
#define ASSIGN(T, x, y) T y = x;
void test54(void) {
  UNINIT(int, a, b);  // expected-warning {{variable 'a' is uninitialized when used here}} \
                      // expected-note {{variable 'a' is declared here}}
  int c;  // expected-note {{initialize the variable 'c' to silence this warning}}
  ASSIGN(int, c, d);  // expected-warning {{variable 'c' is uninitialized when used here}}
}

// Taking the address is fine
struct { struct { void *p; } a; } test55 = { { &test55.a }}; // no-warning
struct { struct { void *p; } a; } test56 = { { &(test56.a) }}; // no-warning

void uninit_in_loop(void) {
  int produce(void);
  void consume(int);
  for (int n = 0; n < 100; ++n) {
    int k; // expected-note {{initialize}}
    consume(k); // expected-warning {{variable 'k' is uninitialized}}
    k = produce();
  }
}

void uninit_in_loop_goto(void) {
  int produce(void);
  void consume(int);
  for (int n = 0; n < 100; ++n) {
    goto skip_decl;
    int k; // expected-note {{initialize}}
skip_decl:
    // FIXME: This should produce the 'is uninitialized' diagnostic, but we
    // don't have enough information in the CFG to easily tell that the
    // variable's scope has been left and re-entered.
    consume(k); // expected-warning {{variable 'k' may be uninitialized}}
    k = produce();
  }
}

typedef char jmp_buf[256];
extern int setjmp(jmp_buf env); // implicitly returns_twice

void do_stuff_and_longjmp(jmp_buf env, int *result) __attribute__((noreturn));

int returns_twice(void) {
  int a; // expected-note {{initialize}}
  if (!a) { // expected-warning {{variable 'a' is uninitialized}}
    jmp_buf env;
    int b;
    if (setjmp(env) == 0) {
      do_stuff_and_longjmp(env, &b);
    } else {
      a = b; // no warning
    }
  }
  return a;
}

int compound_assign(int *arr, int n) {
  int sum; // expected-note {{initialize}}
  for (int i = 0; i < n; ++i)
    sum += arr[i]; // expected-warning {{variable 'sum' is uninitialized}}
  return sum / n;
}

int compound_assign_2(void) {
  int x; // expected-note {{initialize}}
  return x += 1; // expected-warning {{variable 'x' is uninitialized}}
}

int compound_assign_3(void) {
  int x; // expected-note {{initialize}}
  x *= 0; // expected-warning {{variable 'x' is uninitialized}}
  return x;
}

int self_init_in_cond(int *p) {
  int n = ((p && (0 || 1)) && (n = *p)) ? n : -1; // ok
  return n;
}

void test_analyzer_noreturn_aux(void) __attribute__((analyzer_noreturn));

void test_analyzer_noreturn(int y) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
  if (y) {
    test_analyzer_noreturn_aux();
	++x; // no-warning
  }
  else {
	++x; // expected-warning {{variable 'x' is uninitialized when used here}}
  }
}
void test_analyzer_noreturn_2(int y) {
  int x;
  if (y) {
    test_analyzer_noreturn_aux();
  }
  else {
	x = 1;
  }
  ++x; // no-warning
}

// Do not diagnose (functionally) empty structures as being uninitalized
// variables; see GH26842
struct empty {};
struct full_of_empty {
  int : 0;
  int : 12;
  struct empty e;
};

struct empty empty_test_1(void) {
  struct empty e;
  return e; // no-warning
}

struct full_of_empty empty_test_2(void) {
  struct full_of_empty e;
  return e; // no-warning
}
