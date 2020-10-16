// RUN: %clang_cc1 -fsyntax-only -verify %s

#define SA(n, c) int arr##n[(c) ? 1 : -1] = {}

const int AA = 5;

int f1(enum {AA,BB} E) { // expected-warning {{will not be visible outside of this function}}
  SA(1, AA == 0);
  SA(2, BB == 1);
  return BB;
}

int f2(enum {AA=7,BB} E) { // expected-warning {{will not be visible outside of this function}}
  SA(1, AA == 7);
  SA(2, BB == 8);
  return AA;
}

struct a {
};

int f3(struct a { } *); // expected-warning {{will not be visible outside of this function}}

struct A { struct b { int j; } t; }; // expected-note {{previous definition is here}}

int f4(struct A { struct b { int j; } t; } *); // expected-warning {{declaration of 'struct A' will not be visible outside of this function}} expected-warning {{redefinition of 'b' will not be visible outside of this function}}

struct aA {
    struct ab { // expected-note {{previous definition is here}} expected-note {{previous definition is here}}
        int j;
    } b;
};

int f5(struct aA { struct ab { int j; } b; struct ab { char glorx; } glorx; } *); // expected-warning {{declaration of 'struct aA' will not be visible}} expected-warning {{redefinition of 'ab' will not be visible}} expected-warning {{redefinition of 'ab' will not be visible}}

void f6(struct z {int b;} c) { // expected-warning {{declaration of 'struct z' will not be visible outside of this function}}
    struct z d;
    d.b = 4;
}

void pr19018_1 (enum e19018 { qq } x); // expected-warning{{declaration of 'enum e19018' will not be visible outside of this function}}
enum e19018 qq; //expected-error{{tentative definition has type 'enum e19018' that is never completed}} \
                //expected-note{{forward declaration of 'enum e19018'}}

// Only warn once, even if we create two declarations.
void f(struct q *, struct __attribute__((aligned(4))) q *); // expected-warning {{will not be visible outside}}

// This enum inside the function pointer parameter shouldn't leak into the
// function.
enum { BB = 0 };
void enum_in_fun_in_fun(void (*fp)(enum { AA, BB } e)) { // expected-warning {{will not be visible}}
  SA(1, AA == 5); // expected-error {{variable-sized object may not be initialized}}
  SA(2, BB == 0);
}

void f7() {
  extern void ext(struct S { enum E7 { a, b } o; } p); // expected-warning 2 {{will not be visible}}
  ext(a); // expected-error {{use of undeclared identifier}}
}

int f8(struct S { enum E8 { a, b } o; } p) { // expected-warning 2 {{will not be visible}}
  struct S o;
  enum E8 x;
  return a + b;
}
// expected-note@+1 {{forward declaration}}
struct S o; // expected-error {{'struct S' that is never completed}}
// expected-note@+1 {{forward declaration}}
enum E8 x = a + b; // expected-error 2 {{undeclared identifier}} expected-error {{incomplete type 'enum E8'}}

int f9(struct { enum e { a = 1 } b; } c) { // expected-warning {{will not be visible}}
  return a;
}

int f10(
  struct S { // expected-warning {{will not be visible}}
    enum E10 { a, b, c } f; // expected-warning {{will not be visible}}
  } e) {
  return a == b;
}

int f11(
  struct S { // expected-warning {{will not be visible}}
    enum E11 { // expected-warning {{will not be visible}}
      a, b, c
    } // expected-warning {{expected ';' at end of declaration list}}
  } // expected-error {{expected member name or ';'}}
  e);

void f12() {
  extern int ext12(
      struct S12 { } e // expected-warning {{will not be visible}}
      );
  struct S12 o; // expected-error {{incomplete type}} expected-note {{forward declaration}}
}
