// RUN: %clang_cc1 %s -verify -fsyntax-only

int f() __attribute__((deprecated)); // expected-note 2 {{declared here}}
void g() __attribute__((deprecated));
void g(); // expected-note {{declared here}}

extern int var __attribute__((deprecated)); // expected-note {{declared here}}

int a() {
  int (*ptr)() = f; // expected-warning {{'f' is deprecated}}
  f(); // expected-warning {{'f' is deprecated}}

  // test if attributes propagate to functions
  g(); // expected-warning {{'g' is deprecated}}

  return var; // expected-warning {{'var' is deprecated}}
}

// test if attributes propagate to variables
extern int var; // expected-note {{declared here}}
int w() {
  return var; // expected-warning {{'var' is deprecated}}
}

int old_fn() __attribute__ ((deprecated));
int old_fn(); // expected-note {{declared here}}
int (*fn_ptr)() = old_fn; // expected-warning {{'old_fn' is deprecated}}

int old_fn() {
  return old_fn()+1;  // no warning, deprecated functions can use deprecated symbols.
}


struct foo {
  int x __attribute__((deprecated)); // expected-note 3 {{declared here}}
};

void test1(struct foo *F) {
  ++F->x;  // expected-warning {{'x' is deprecated}}
  struct foo f1 = { .x = 17 }; // expected-warning {{'x' is deprecated}}
  struct foo f2 = { 17 }; // expected-warning {{'x' is deprecated}}
}

typedef struct foo foo_dep __attribute__((deprecated)); // expected-note 12 {{declared here}}
foo_dep *test2;    // expected-warning {{'foo_dep' is deprecated}}

struct __attribute__((deprecated, 
                      invalid_attribute)) bar_dep ;  // expected-warning {{unknown attribute 'invalid_attribute' ignored}} expected-note 2 {{declared here}}

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
enum __attribute__((deprecated)) Test20 { // expected-note {{declared here}}
  test20_a __attribute__((deprecated)), // expected-note {{declared here}}
  test20_b // expected-note {{declared here}}
};
void test20() {
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
