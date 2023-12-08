// RUN: %clang_cc1 -Wmissing-variable-declarations -fsyntax-only -verify %s

int vbad1; // expected-warning{{no previous extern declaration for non-static variable 'vbad1'}}
// expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}

int vbad2;
int vbad2 = 10; // expected-warning{{no previous extern declaration for non-static variable 'vbad2'}}
// expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}

struct { // expected-note{{declare 'static' if the variable is not intended to be used outside of this translation unit}}
  int mgood1;
} vbad3; // expected-warning{{no previous extern declaration for non-static variable 'vbad3'}}

int vbad4;
int vbad4 = 10; // expected-warning{{no previous extern declaration for non-static variable 'vbad4'}}
// expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}
extern int vbad4;

extern int vgood1;
int vgood1;
int vgood1 = 10;
