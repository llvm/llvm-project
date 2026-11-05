// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -analyzer-checker=core,unix.Malloc,debug.ExprInspection -DNO_CROSSCHECK -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config crosscheck-with-z3=true -verify %s
// REQUIRES: z3

void clang_analyzer_dump(float);

// The built-in range-based constraint manager reasons about each `%` / `/`
// sub-expression as an independent, coarsely over-approximated symbol and
// cannot relate two of them, so it wrongly enters these contradictory
// branches and reports a null dereference.  Z3, with exact modular/division
// semantics, refutes the path -- which is exactly what the cross-check is for.

int rem_parity(int x) // `x` even and `x + 1` even: impossible
{
  int *z = 0;
  if (x % 2 == 0)
    if ((x + 1) % 2 == 0)
#ifdef NO_CROSSCHECK
      return *z; // expected-warning {{Dereference of null pointer (loaded from variable 'z')}}
#else
      return *z; // no-warning
#endif
  return 0;
}

int mul_parity(int x, int y) // `x == 2 * y` is even, yet assumed odd: impossible
{
  int *z = 0;
  if (x == 2 * y)
    if (x % 2 != 0)
#ifdef NO_CROSSCHECK
      return *z; // expected-warning {{Dereference of null pointer (loaded from variable 'z')}}
#else
      return *z; // no-warning
#endif
  return 0;
}

int div_rem_identity(int x) // `(x / 10) * 10 + x % 10 == x` always holds
{
  int *z = 0;
  if (x > 0 && x < 1000)
    if ((x / 10) * 10 + (x % 10) != x)
#ifdef NO_CROSSCHECK
      return *z; // expected-warning {{Dereference of null pointer (loaded from variable 'z')}}
#else
      return *z; // no-warning
#endif
  return 0;
}

void g(int d);

void f(int *a, int *b) {
  int c = 5;
  if ((a - b) == 0)
    c = 0;
  if (a != b)
    g(3 / c); // no-warning
}

_Bool nondet_bool();

void h(int d) {
  int x, y, k, z = 1;
  while (z < k) { // expected-warning {{The right operand of '<' is a garbage value}}
    z = 2 * z;
  }
}

void i() {
  _Bool c = nondet_bool();
  if (c) {
    h(1);
  } else {
    h(2);
  }
}

void floatUnaryNegInEq(int h, int l) {
  int j;
  clang_analyzer_dump(-(float)h); // expected-warning-re{{-(float) (reg_${{[0-9]+}}<int h>)}}
  clang_analyzer_dump((float)l); // expected-warning-re {{(float) (reg_${{[0-9]+}}<int l>)}}
  if (-(float)h != (float)l) {  // should not crash
    j += 10;
    // expected-warning@-1{{The left expression of the compound assignment uses uninitialized memory [core.uninitialized.Assign]}}
  }
}

void floatUnaryLNotInEq(int h, int l) {
  int j;
  clang_analyzer_dump(!(float)h); // expected-warning{{Unknown}}
  clang_analyzer_dump((float)l); // expected-warning-re {{(float) (reg_${{[0-9]+}}<int l>)}}
  if ((!(float)h) != (float)l) {  // should not crash
    j += 10;
    // expected-warning@-1{{The left expression of the compound assignment uses uninitialized memory [core.uninitialized.Assign]}}
  }
}

// don't crash, and also produce a core.CallAndMessage finding
void a(int);
typedef struct {
  int b;
} c;
c *d;
void e() {
  (void)d->b;
  int f;
  a(f); // expected-warning {{1st function call argument is an uninitialized value [core.CallAndMessage]}}
}

void nullDerefGuardedByAtomicComp(int input) {
  int *nullPointer = 0;
  _Atomic int atomicValue = input;
  if (atomicValue == 0) {
    *nullPointer = 1; // no-crash // expected-warning {{Dereference of null pointer (loaded from variable 'nullPointer')}}
  }
}
