// RUN: %clang_cc1 -verify -fsyntax-only -Wno-private-extern %s

static int g0; // expected-note{{previous definition}}
int g0; // expected-error{{non-static declaration of 'g0' follows static declaration}}

static int g1;
extern int g1;

static int g2; 
__private_extern__ int g2;

int g3; // expected-note{{previous definition}}
static int g3; // expected-error{{static declaration of 'g3' follows non-static declaration}}

extern int g4; // expected-note{{previous definition}}
static int g4; // expected-error{{static declaration of 'g4' follows non-static declaration}}

__private_extern__ int g5; // expected-note{{previous definition}}
static int g5; // expected-error{{static declaration of 'g5' follows non-static declaration}}

void f0() {
  int g6; // expected-note {{previous}}
  extern int g6; // expected-error {{extern declaration of 'g6' follows non-extern declaration}}
}

void f1() {
  int g7; // expected-note {{previous}}
  __private_extern__ int g7; // expected-error {{extern declaration of 'g7' follows non-extern declaration}}
}

void f2() {
  extern int g8; // expected-note{{previous definition}}
  int g8; // expected-error {{non-extern declaration of 'g8' follows extern declaration}}
}

void f3() {
  __private_extern__ int g9; // expected-note{{previous definition}}
  int g9; // expected-error {{non-extern declaration of 'g9' follows extern declaration}}
}

void f4() {
  extern int g10;
  extern int g10;
}

void f5() {
  __private_extern__ int g11;
  __private_extern__ int g11;
}

void f6() {
  // FIXME: Diagnose
  extern int g12;
  __private_extern__ int g12;
}

void f7() {
  // FIXME: Diagnose
  __private_extern__ int g13;
  extern int g13;
}

struct s0;
void f8() {
  extern struct s0 g14;
  __private_extern__ struct s0 g14;
}
struct s0 { int x; };

void f9() {
  extern int g15 = 0; // expected-error{{'extern' variable cannot have an initializer}}
  // FIXME: linkage specifier in warning.
  __private_extern__ int g16 = 0; // expected-error{{'extern' variable cannot have an initializer}}
}

extern int g17;
int g17 = 0;

extern int g18 = 0; // expected-warning{{'extern' variable has an initializer}}

__private_extern__ int g19;
int g19 = 0;

__private_extern__ int g20 = 0;
