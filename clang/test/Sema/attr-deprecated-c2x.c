// RUN: %clang_cc1 %s -verify -fsyntax-only --std=c2x

int f() [[deprecated]]; // expected-note 2 {{'f' has been explicitly marked deprecated here}}
void g() [[deprecated]];// expected-note {{'g' has been explicitly marked deprecated here}}
void g();

extern int var [[deprecated]]; // expected-note 2 {{'var' has been explicitly marked deprecated here}}

int a() {
  int (*ptr)() = f; // expected-warning {{'f' is deprecated}}
  f(); // expected-warning {{'f' is deprecated}}

  // test if attributes propagate to functions
  g(); // expected-warning {{'g' is deprecated}}

  return var; // expected-warning {{'var' is deprecated}}
}

// test if attributes propagate to variables
extern int var;
int w() {
  return var; // expected-warning {{'var' is deprecated}}
}

int old_fn() [[deprecated]];// expected-note {{'old_fn' has been explicitly marked deprecated here}}
int old_fn();
int (*fn_ptr)() = old_fn; // expected-warning {{'old_fn' is deprecated}}

int old_fn() {
  return old_fn()+1;  // no warning, deprecated functions can use deprecated symbols.
}

struct foo {
  int x [[deprecated]]; // expected-note 3 {{'x' has been explicitly marked deprecated here}}
};

void test1(struct foo *F) {
  ++F->x;  // expected-warning {{'x' is deprecated}}
  struct foo f1 = { .x = 17 }; // expected-warning {{'x' is deprecated}}
  struct foo f2 = { 17 }; // expected-warning {{'x' is deprecated}}
}

typedef struct foo foo_dep [[deprecated]]; // expected-note {{'foo_dep' has been explicitly marked deprecated here}}
foo_dep *test2;    // expected-warning {{'foo_dep' is deprecated}}

struct [[deprecated, // expected-note {{'bar_dep' has been explicitly marked deprecated here}}
         invalid_attribute]] bar_dep ;  // expected-warning {{unknown attribute 'invalid_attribute' ignored}}

struct bar_dep *test3;   // expected-warning {{'bar_dep' is deprecated}}

[[deprecated("this is the message")]] int i; // expected-note {{'i' has been explicitly marked deprecated here}}
void test4(void) {
  i = 12; // expected-warning {{'i' is deprecated: this is the message}}
}
