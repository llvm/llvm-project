// RUN: %clang_cc1 -fsyntax-only -verify %s

struct foo; // expected-note 5 {{forward declaration of 'struct foo'}}

void b;  // expected-error {{variable has incomplete type 'void'}}
struct foo f; // expected-error{{tentative definition has type 'struct foo' that is never completed}}

static void c; // expected-error {{variable has incomplete type 'void'}}
static struct foo g;  // expected-warning {{tentative definition of variable with internal linkage has incomplete non-array type 'struct foo'}} \
    expected-error{{tentative definition has type 'struct foo' that is never completed}}

extern void d;
extern struct foo e;

int ary[]; // expected-warning {{tentative array definition assumed to have one element}}
struct foo bary[]; // expected-error {{array has incomplete element type 'struct foo'}}

void func(void) {
  int ary[]; // expected-error{{definition of variable with array type needs an explicit size or an initializer}}
  void b; // expected-error {{variable has incomplete type 'void'}}
  struct foo f; // expected-error {{variable has incomplete type 'struct foo'}}
}

int h[]; // expected-warning {{tentative array definition assumed to have one element}}
int (*i)[] = &h+1; // expected-error {{arithmetic on a pointer to an incomplete type 'int[]'}}

struct bar j = {1}; // expected-error {{variable has incomplete type 'struct bar'}} \
    expected-note {{forward declaration of 'struct bar'}}
struct bar k;
struct bar { int a; };

