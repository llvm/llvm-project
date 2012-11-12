// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

// GCC will accept anything as the argument of weakref. Should we
// check for an existing decl?
static int a1() __attribute__((weakref ("foo")));
static int a2() __attribute__((weakref, alias ("foo")));

static int a3 __attribute__((weakref ("foo")));
static int a4 __attribute__((weakref, alias ("foo")));

// gcc rejects, clang accepts
static int a5 __attribute__((alias ("foo"), weakref));

// this is pointless, but accepted by gcc. We reject it.
static int a6 __attribute__((weakref)); //expected-error {{weakref declaration of 'a6' must also have an alias attribute}}

// gcc warns, clang rejects
void f(void) {
  static int a __attribute__((weakref ("v2"))); // expected-error {{declaration of 'a' must be in a global context}}
}

// both gcc and clang reject
class c {
  static int a __attribute__((weakref ("v2"))); // expected-error {{declaration of 'a' must be in a global context}}
  static int b() __attribute__((weakref ("f3"))); // expected-error {{declaration of 'b' must be in a global context}}
};
int a7() __attribute__((weakref ("f1"))); // expected-error {{weakref declaration must have internal linkage}}
int a8 __attribute__((weakref ("v1"))); // expected-error {{weakref declaration must have internal linkage}}

// gcc accepts this
int a9 __attribute__((weakref)); // expected-error {{weakref declaration must have internal linkage}}
