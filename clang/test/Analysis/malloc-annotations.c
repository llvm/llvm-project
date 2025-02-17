// RUN: %clang_analyze_cc1 -verify \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-checker=alpha.core.CastSize \
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
  free(p); // expected-warning{{Attempt to free released memory}}
}

void f2_realloc_0(void) {
  int *p = malloc(12);
  realloc(p,0);
  realloc(p,0); // expected-warning{{Attempt to free released memory}}
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
  free(p); // expected-warning{{Attempt to free released memory}}
}

void af2b(void) {
  int *p = my_malloc(12);
  free(p);
  my_free(p); // expected-warning{{Attempt to free released memory}}
}

void af2c(void) {
  int *p = my_malloc(12);
  free(p);
  my_hold(p); // expected-warning{{Attempt to free released memory}}
}

void af2d(void) {
  int *p = my_malloc(12);
  free(p);
  my_hold2(0, 0, p); // expected-warning{{Attempt to free released memory}}
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
  free(p); // expected-warning{{Attempt to free non-owned memory}}
}

int * af4(void) {
  int *p = my_malloc(12);
  my_free(p);
  return p; // expected-warning{{Use of memory after it is freed}}
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
  x[0] = 'a'; // expected-warning{{Use of memory after it is freed}}
}

void f7_realloc(void) {
  char *x = (char*) malloc(4);
  realloc(x,0);
  x[0] = 'a'; // expected-warning{{Use of memory after it is freed}}
}

void PR6123(void) {
  int *x = malloc(11); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
}

void PR7217(void) {
  int *buf = malloc(2); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  buf[1] = 'c'; // not crash
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
  char result = buf[1]; // expected-warning{{undefined}}
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

