// RUN: %clang_cc1 -fsyntax-only -Wunused-but-set-variable -verify %s

static int set_unused; // expected-warning {{variable 'set_unused' set but not used}}
static int set_and_used;
static int only_used;
static int addr_taken;
extern int external_var;  // no warning (external linkage)
extern int global_var;  // no warning (not static)

void f1() {
  set_unused = 1;
  set_and_used = 2;

  int x = set_and_used;
  (void)x;

  int y = only_used;
  (void)y;

  int *p = &addr_taken;
  (void)p;

  external_var = 3;
  global_var = 4;
}

// test across multiple functions
static int set_used1;
static int set_used2;

static int set1; // expected-warning {{variable 'set1' set but not used}}
static int set2; // expected-warning {{variable 'set2' set but not used}}

void f2() {
  set1 = 1;
  set_used1 = 1;

  int x = set_used2;
  (void)x;
}

void f3() {
  set2 = 2;
  set_used2 = 2;

  int x = set_used1;
  (void)x;
}

static volatile int vol_set; // expected-warning {{variable 'vol_set' set but not used}}
void f4() {
  vol_set = 1;
}

// read and use
static int compound; // expected-warning{{variable 'compound' set but not used}}
static volatile int vol_compound;
static int unary; // expected-warning{{variable 'unary' set but not used}}
static volatile int vol_unary;
void f5() {
  compound += 1;
  vol_compound += 1;
  unary++;
  vol_unary++;
}

struct S {
  int i;
};
static struct S s_set;  // expected-warning{{variable 's_set' set but not used}}
static struct S s_used;
void f6() {
  struct S t;
  s_set = t;
  t = s_used;
}

// multiple assignments
static int multi; // expected-warning{{variable 'multi' set but not used}}
void f7() {
  multi = 1;
  multi = 2;
  multi = 3;
}
