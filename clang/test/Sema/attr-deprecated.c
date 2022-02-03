// RUN: %clang_cc1 %s -verify -fsyntax-only
// RUN: %clang_cc1 %s -std=c89 -verify -fsyntax-only
// RUN: %clang_cc1 %s -std=c99 -verify -fsyntax-only

int f(void) __attribute__((deprecated)); // expected-note 2 {{'f' has been explicitly marked deprecated here}}
void g(void) __attribute__((deprecated));// expected-note {{'g' has been explicitly marked deprecated here}}
void g(void); 

extern int var __attribute__((deprecated)); // expected-note 2 {{'var' has been explicitly marked deprecated here}}

int a(void) {
  int (*ptr)(void) = f; // expected-warning {{'f' is deprecated}}
  f(); // expected-warning {{'f' is deprecated}}

  // test if attributes propagate to functions
  g(); // expected-warning {{'g' is deprecated}}

  return var; // expected-warning {{'var' is deprecated}}
}

// test if attributes propagate to variables
extern int var; 
int w(void) {
  return var; // expected-warning {{'var' is deprecated}}
}

int old_fn(void) __attribute__ ((deprecated));// expected-note {{'old_fn' has been explicitly marked deprecated here}}
int old_fn(void); 
int (*fn_ptr)(void) = old_fn; // expected-warning {{'old_fn' is deprecated}}

int old_fn(void) {
  return old_fn()+1;  // no warning, deprecated functions can use deprecated symbols.
}


struct foo {
  int x __attribute__((deprecated)); // expected-note 3 {{'x' has been explicitly marked deprecated here}}
};

void test1(struct foo *F) {
  ++F->x;  // expected-warning {{'x' is deprecated}}
  struct foo f1 = { .x = 17 }; // expected-warning {{'x' is deprecated}}
  struct foo f2 = { 17 }; // expected-warning {{'x' is deprecated}}
}

typedef struct foo foo_dep __attribute__((deprecated)); // expected-note 12 {{'foo_dep' has been explicitly marked deprecated here}}
foo_dep *test2;    // expected-warning {{'foo_dep' is deprecated}}

struct __attribute__((deprecated, // expected-note 2 {{'bar_dep' has been explicitly marked deprecated here}}
                      invalid_attribute)) bar_dep ;  // expected-warning {{unknown attribute 'invalid_attribute' ignored}} 

struct bar_dep *test3;   // expected-warning {{'bar_dep' is deprecated}}


// These should not warn because the actually declaration itself is deprecated.
// rdar://6756623
foo_dep *test4 __attribute__((deprecated));
struct bar_dep *test5 __attribute__((deprecated));

typedef foo_dep test6(struct bar_dep*); // expected-warning {{'foo_dep' is deprecated}} \
                                        // expected-warning {{'bar_dep' is deprecated}}
typedef foo_dep test7(struct bar_dep*) __attribute__((deprecated));

int test8(char *p) {
  p += sizeof(foo_dep); // expected-warning {{'foo_dep' is deprecated}}

  foo_dep *ptr;         // expected-warning {{'foo_dep' is deprecated}}
  ptr = (foo_dep*) p;   // expected-warning {{'foo_dep' is deprecated}}

  int func(foo_dep *foo); // expected-warning {{'foo_dep' is deprecated}}
  return func(ptr);
}

foo_dep *test9(void) __attribute__((deprecated));
foo_dep *test9(void) {
  void* myalloc(unsigned long);

  foo_dep *ptr
    = (foo_dep*)
        myalloc(sizeof(foo_dep));
  return ptr;
}

void test10(void) __attribute__((deprecated));
void test10(void) {
  if (sizeof(foo_dep) == sizeof(void*)) {
  }
  foo_dep *localfunc(void);
  foo_dep localvar;
}

char test11[sizeof(foo_dep)] __attribute__((deprecated));
char test12[sizeof(foo_dep)]; // expected-warning {{'foo_dep' is deprecated}}

int test13(foo_dep *foo) __attribute__((deprecated));
int test14(foo_dep *foo); // expected-warning {{'foo_dep' is deprecated}}

unsigned long test15 = sizeof(foo_dep); // expected-warning {{'foo_dep' is deprecated}}
unsigned long test16 __attribute__((deprecated))
  = sizeof(foo_dep);

foo_dep test17, // expected-warning {{'foo_dep' is deprecated}}
        test18 __attribute__((deprecated)),
        test19;

// rdar://problem/8518751
enum __attribute__((deprecated)) Test20 { // expected-note 2 {{'Test20' has been explicitly marked deprecated here}}
  test20_a __attribute__((deprecated)), // expected-note {{'test20_a' has been explicitly marked deprecated here}}
  test20_b
};
void test20(void) {
  enum Test20 f; // expected-warning {{'Test20' is deprecated}}
  f = test20_a; // expected-warning {{'test20_a' is deprecated}}
  f = test20_b; // expected-warning {{'test20_b' is deprecated}}
}

char test21[__has_feature(attribute_deprecated_with_message) ? 1 : -1];

struct test22 {
  foo_dep a __attribute((deprecated));
  foo_dep b; // expected-warning {{'foo_dep' is deprecated}}
  foo_dep c, d __attribute((deprecated)); // expected-warning {{'foo_dep' is deprecated}}
  __attribute((deprecated)) foo_dep e, f;
};

typedef int test23_ty __attribute((deprecated)); 
// Redefining a typedef is a C11 feature.
#if __STDC_VERSION__ <= 199901L
// expected-note@-3 {{'test23_ty' has been explicitly marked deprecated here}}
#else
// expected-note@-5 {{'test23_ty' has been explicitly marked deprecated here}}
typedef int test23_ty; 
#endif
test23_ty test23_v; // expected-warning {{'test23_ty' is deprecated}}
