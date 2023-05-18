// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -analyze -analyzer-checker=core,unix.Malloc,debug.ExprInspection -DNO_CROSSCHECK -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -analyze -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config crosscheck-with-z3=true -verify %s
// REQUIRES: z3

void clang_analyzer_dump(float);

int foo(int x) 
{
  int *z = 0;
  if ((x & 1) && ((x & 1) ^ 1))
#ifdef NO_CROSSCHECK
      return *z; // expected-warning {{Dereference of null pointer (loaded from variable 'z')}}
#else
      return *z; // no-warning
#endif
  return 0;
}

int unary(int x, long l)
{
  int *z = 0;
  int y = l;
  if ((x & 1) && ((x & 1) ^ 1))
    if (-y)
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
    // expected-warning@-1{{garbage}}
  }
}

void floatUnaryLNotInEq(int h, int l) {
  int j;
  clang_analyzer_dump(!(float)h); // expected-warning{{Unknown}}
  clang_analyzer_dump((float)l); // expected-warning-re {{(float) (reg_${{[0-9]+}}<int l>)}}
  if ((!(float)h) != (float)l) {  // should not crash
    j += 10;
    // expected-warning@-1{{garbage}}
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
