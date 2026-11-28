// RUN: %clang_cc1 -fsyntax-only -verify -Wno-strict-prototypes %s

static int a16[];  // expected-warning {{tentative array definition assumed to have one element}}

void f16(void) {
    extern int a16[];
}


// PR10013: Scope of extern declarations extend past enclosing block
extern int PR10013_x;
int PR10013(void) {
  int *PR10013_x = 0;
  {
    extern int PR10013_x;
    extern int PR10013_x;
  }

  return PR10013_x; // expected-error{{incompatible pointer to integer conversion}}
}

static int test1_a[]; // expected-warning {{tentative array definition assumed to have one element}}
extern int test1_a[];

void test2declarer(void) { extern int test2_array[100]; }
extern int test2_array[];
int test2v = sizeof(test2_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int[]'}}

void test3declarer(void) {
  { extern int test3_array[100]; }
  extern int test3_array[];
  int x = sizeof(test3_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int[]'}}
}

void test4(void) {
  extern int test4_array[];
  {
    extern int test4_array[100];
    int x = sizeof(test4_array); // fine
  }
  int x = sizeof(test4_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int[]'}}
}

// Test that invalid local extern declarations of library
// builtins behave reasonably.
extern void abort(void); // expected-note 2 {{previous declaration is here}}
extern float *calloc(void); // expected-warning {{incompatible redeclaration of library function}} expected-note {{is a builtin}} expected-note 2 {{previous declaration is here}}
void test5a(void) {
  int abort(void); // expected-error {{conflicting types}}
  float *malloc(void); // expected-warning {{incompatible redeclaration of library function}} expected-note 2 {{is a builtin}}
  int *calloc(void); // expected-error {{conflicting types}}
}
void test5b(void) {
  int abort(void); // expected-error {{conflicting types}}
  float *malloc(void); // expected-warning {{incompatible redeclaration of library function}}
  int *calloc(void); // expected-error {{conflicting types}}
}
void test5c(void) {
  void (*_abort)(void) = &abort;
  void *(*_malloc)() = &malloc;
  float *(*_calloc)() = &calloc;
}

void test6(void) {
  extern int test6_array1[100];
  extern int test6_array2[100];
  void test6_fn1(int*);
  void test6_fn2(int*);
  {
    // Types are only merged from visible declarations.
    char test6_array2;
    char test6_fn2;
    {
      extern int test6_array1[];
      extern int test6_array2[];
      (void)sizeof(test6_array1); // ok
      (void)sizeof(test6_array2); // expected-error {{incomplete type}}

      void test6_fn1();
      void test6_fn2();
      test6_fn1(1.2); // expected-error {{passing 'double' to parameter of incompatible type 'int *'}}
      // FIXME: This is valid, but we should warn on it.
      test6_fn2(1.2);
    }
  }
}

// GH242842: tentative array completed by later extern should not warn.
int gh242842_a[];
extern int gh242842_a[10];

static int gh242842_b[];
extern int gh242842_b[10];

int gh242842_c[];
extern int gh242842_c[10];
extern int gh242842_c[];

// Block-scope extern does not complete file-scope tentative definition.
static int gh242842_d[]; // expected-warning {{tentative array definition assumed to have one element}}
void gh242842_fn(void) {
  extern int gh242842_d[10];
}
