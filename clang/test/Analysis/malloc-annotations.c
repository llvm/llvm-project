// RUN: %clang_analyze_cc1 -verify \
// RUN:   -Wno-alloc-size \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config unix.DynamicMemoryModeling:Optimistic=true %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
void __attribute((ownership_returns(malloc))) *my_malloc(size_t);
void __attribute((ownership_takes(malloc, 1))) my_free(void *);
void my_freeBoth(void *, void *)
       __attribute((ownership_holds(malloc, 1, 2)));
void __attribute((ownership_returns(malloc, 1))) *my_malloc2(size_t);
void __attribute((ownership_holds(malloc, 1))) my_hold(void *);

// Duplicate attributes are silly, but not an error.
// Duplicate attribute has no extra effect.
// If two are of different kinds, that is an error and reported as such.
void __attribute((ownership_holds(malloc, 1)))
__attribute((ownership_holds(malloc, 1)))
__attribute((ownership_holds(malloc, 3))) my_hold2(void *, void *, void *);

__attribute((ownership_returns(user_malloc, 1))) void *user_malloc(size_t);
__attribute((ownership_takes(user_malloc, 1))) void user_free(void *);

void clang_analyzer_dump(int);

void *my_malloc3(size_t);
void *myglobalpointer;
struct stuff {
  void *somefield;
};
struct stuff myglobalstuff;

void f1(void) {
  int *p = malloc(12);
  return; // expected-warning{{Potential leak of memory pointed to by}}
}

void f2(void) {
  int *p = malloc(12);
  free(p);
  free(p); // expected-warning{{Attempt to release already released memory}}
}

void f2_realloc_0(void) {
  int *p = malloc(12);
  realloc(p,0);
  realloc(p,0); // expected-warning{{Attempt to release already released memory}}
}

void f2_realloc_1(void) {
  int *p = malloc(12);
  int *q = realloc(p,0); // no-warning
}

// ownership attributes tests
void naf1(void) {
  int *p = my_malloc3(12);
  return; // no-warning
}

void n2af1(void) {
  int *p = my_malloc2(12);
  return; // expected-warning{{Potential leak of memory pointed to by}}
}

void af1(void) {
  int *p = my_malloc(12);
  return; // expected-warning{{Potential leak of memory pointed to by}}
}

void af1_b(void) {
  int *p = my_malloc(12);
} // expected-warning{{Potential leak of memory pointed to by}}

void af1_c(void) {
  myglobalpointer = my_malloc(12); // no-warning
}

void af1_d(void) {
  struct stuff mystuff;
  mystuff.somefield = my_malloc(12);
} // expected-warning{{Potential leak of memory pointed to by}}

// Test that we can pass out allocated memory via pointer-to-pointer.
void af1_e(void **pp) {
  *pp = my_malloc(42); // no-warning
}

void af1_f(struct stuff *somestuff) {
  somestuff->somefield = my_malloc(12); // no-warning
}

// Allocating memory for a field via multiple indirections to our arguments is OK.
void af1_g(struct stuff **pps) {
  *pps = my_malloc(sizeof(struct stuff)); // no-warning
  (*pps)->somefield = my_malloc(42); // no-warning
}

void af2(void) {
  int *p = my_malloc(12);
  my_free(p);
  free(p); // expected-warning{{Attempt to release already released memory}}
}

void af2b(void) {
  int *p = my_malloc(12);
  free(p);
  my_free(p); // expected-warning{{Attempt to release already released memory}}
}

void af2c(void) {
  int *p = my_malloc(12);
  free(p);
  my_hold(p); // expected-warning{{Attempt to release already released memory}}
}

void af2d(void) {
  int *p = my_malloc(12);
  free(p);
  my_hold2(0, 0, p); // expected-warning{{Attempt to release already released memory}}
}

// No leak if malloc returns null.
void af2e(void) {
  int *p = my_malloc(12);
  if (!p)
    return; // no-warning
  free(p); // no-warning
}

// This case inflicts a possible double-free.
void af3(void) {
  int *p = my_malloc(12);
  my_hold(p);
  free(p); // expected-warning{{Attempt to release non-owned memory}}
}

int * af4(void) {
  int *p = my_malloc(12);
  my_free(p);
  return p; // expected-warning{{Use of memory after it is released}}
}

// This case is (possibly) ok, be conservative
int * af5(void) {
  int *p = my_malloc(12);
  my_hold(p);
  return p; // no-warning
}



// This case tests that storing malloc'ed memory to a static variable which is
// then returned is not leaked.  In the absence of known contracts for functions
// or inter-procedural analysis, this is a conservative answer.
int *f3(void) {
  static int *p = 0;
  p = malloc(12);
  return p; // no-warning
}

// This case tests that storing malloc'ed memory to a static global variable
// which is then returned is not leaked.  In the absence of known contracts for
// functions or inter-procedural analysis, this is a conservative answer.
static int *p_f4 = 0;
int *f4(void) {
  p_f4 = malloc(12);
  return p_f4; // no-warning
}

int *f5(void) {
  int *q = malloc(12);
  q = realloc(q, 20);
  return q; // no-warning
}

void f6(void) {
  int *p = malloc(12);
  if (!p)
    return; // no-warning
  else
    free(p);
}

void f6_realloc(void) {
  int *p = malloc(12);
  if (!p)
    return; // no-warning
  else
    realloc(p,0);
}


char *doit2(void);
void pr6069(void) {
  char *buf = doit2();
  free(buf);
}

void pr6293(void) {
  free(0);
}

void f7(void) {
  char *x = (char*) malloc(4);
  free(x);
  x[0] = 'a'; // expected-warning{{Use of memory after it is released}}
}

void f7_realloc(void) {
  char *x = (char*) malloc(4);
  realloc(x,0);
  x[0] = 'a'; // expected-warning{{Use of memory after it is released}}
}

void mallocCastToVoid(void) {
  void *p = malloc(2);
  const void *cp = p; // not crash
  free(p);
}

void mallocCastToFP(void) {
  void *p = malloc(2);
  void (*fp)(void) = p; // not crash
  free(p);
}

// This tests that malloc() buffers are undefined by default
char mallocGarbage (void) {
  char *buf = malloc(2);
  char result = buf[1]; // expected-warning{{uninitialized}}
  free(buf);
  return result;
}

// This tests that calloc() buffers need to be freed
void callocNoFree (void) {
  char *buf = calloc(2,2);
  return; // expected-warning{{Potential leak of memory pointed to by}}
}

// These test that calloc() buffers are zeroed by default
char callocZeroesGood (void) {
  char *buf = calloc(2,2);
  char result = buf[3]; // no-warning
  if (buf[1] == 0) {
    free(buf);
  }
  return result; // no-warning
}

char callocZeroesBad (void) {
  char *buf = calloc(2,2);
  char result = buf[3]; // no-warning
  if (buf[1] != 0) {
    free(buf); // expected-warning{{never executed}}
  }
  return result; // expected-warning{{Potential leak of memory pointed to by}}
}

void testMultipleFreeAnnotations(void) {
  int *p = malloc(12);
  int *q = malloc(12);
  my_freeBoth(p, q);
}

void testNoUninitAttr(void) {
  int *p = user_malloc(sizeof(int));
  int read = p[0]; // no-warning
  clang_analyzer_dump(p[0]); // expected-warning{{Unknown}}
  user_free(p);
}

// Regression test for GH#183344 — crash when a function has both
// ownership_returns and ownership_takes attributes.
typedef struct GH183344_X GH183344_X;
typedef struct GH183344_Y GH183344_Y;

GH183344_Y *GH183344_X_to_Y(GH183344_X *x)
    __attribute__((ownership_returns(GH183344_Y)))
    __attribute__((ownership_takes(GH183344_X, 1)));

void testGH183344(void) {
  GH183344_Y *y = GH183344_X_to_Y(0); // no-crash
  (void)y;
} // expected-warning{{Potential leak of memory pointed to by 'y'}}

// Extended regression tests for GH#183344 — additional combinations of
// ownership_returns, ownership_takes, and ownership_holds.

GH183344_X *GH183344_alloc_X(void)
    __attribute__((ownership_returns(GH183344_X)));
void GH183344_free_X(GH183344_X *)
    __attribute__((ownership_takes(GH183344_X, 1)));
void GH183344_free_Y(GH183344_Y *)
    __attribute__((ownership_takes(GH183344_Y, 1)));

// Returns + Holds variant: Y is allocated, X is held (not consumed) by callee.
GH183344_Y *GH183344_X_to_Y_hold(GH183344_X *x)
    __attribute__((ownership_returns(GH183344_Y)))
    __attribute__((ownership_holds(GH183344_X, 1)));

// Returns + two Takes (same pool): both X arguments are consumed, Y is
// returned. Multiple arg indices in one attribute (same pool) is valid;
// two ownership_takes attributes with *different* pool names are not.
GH183344_Y *GH183344_combine_XX(GH183344_X *x1, GH183344_X *x2)
    __attribute__((ownership_returns(GH183344_Y)))
    __attribute__((ownership_takes(GH183344_X, 1, 2)));

// No-crash for Returns+Holds with null input — same crash pattern as the
// original GH183344 bug but with ownership_holds instead of ownership_takes.
void testGH183344_ReturnsHolds_NullInput(void) {
  GH183344_Y *y = GH183344_X_to_Y_hold(0); // no-crash
  (void)y;
} // expected-warning{{Potential leak of memory pointed to by 'y'}}

// Returns+Takes: allocate X, convert to Y (X consumed), free Y — no warnings.
void testGH183344_ReturnsTakes_CleanUse(void) {
  GH183344_X *x = GH183344_alloc_X();
  GH183344_Y *y = GH183344_X_to_Y(x);
  GH183344_free_Y(y);
} // no-warning

// Returns+Takes: after the conversion X is consumed; freeing it again is a
// double-free.
void testGH183344_ReturnsTakes_DoubleFreeInput(void) {
  GH183344_X *x = GH183344_alloc_X();
  GH183344_Y *y = GH183344_X_to_Y(x);
  GH183344_free_X(x); // expected-warning{{Attempt to release already released memory}}
  GH183344_free_Y(y);
}

// Returns+Takes: X is consumed but Y is never freed — leak on Y.
void testGH183344_ReturnsTakes_LeakRetval(void) {
  GH183344_X *x = GH183344_alloc_X();
  GH183344_Y *y = GH183344_X_to_Y(x);
  (void)y;
} // expected-warning{{Potential leak of memory pointed to by 'y'}}

// Returns+Holds: after the hold, X is non-owned by the caller; freeing it
// produces a "non-owned memory" warning (analogous to af3).
void testGH183344_ReturnsHolds_FreeHeldInput(void) {
  GH183344_X *x = GH183344_alloc_X();
  GH183344_Y *y = GH183344_X_to_Y_hold(x);
  GH183344_free_X(x); // expected-warning{{Attempt to release non-owned memory}}
  GH183344_free_Y(y);
}

// Multiple Takes (same pool) + Returns: both X inputs are consumed, Y is
// returned and freed — no warnings.
void testGH183344_CombineXX_CleanUse(void) {
  GH183344_X *x1 = GH183344_alloc_X();
  GH183344_X *x2 = GH183344_alloc_X();
  GH183344_Y *y = GH183344_combine_XX(x1, x2);
  GH183344_free_Y(y);
} // no-warning

// Multiple Takes (same pool): after the combine, x2 is already consumed;
// freeing it again is a double-free.
void testGH183344_CombineXX_DoubleFreeSecondInput(void) {
  GH183344_X *x1 = GH183344_alloc_X();
  GH183344_X *x2 = GH183344_alloc_X();
  GH183344_Y *y = GH183344_combine_XX(x1, x2);
  GH183344_free_X(x2); // expected-warning{{Attempt to release already released memory}}
  GH183344_free_Y(y);
}
