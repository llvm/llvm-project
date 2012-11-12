// RUN: %clang -Wmissing-variable-declarations -fsyntax-only -Xclang -verify %s

int vbad1; // expected-warning{{no previous extern declaration for non-static variable 'vbad1'}}

int vbad2;
int vbad2 = 10; // expected-warning{{no previous extern declaration for non-static variable 'vbad2'}}

struct {
  int mgood1;
} vbad3; // expected-warning{{no previous extern declaration for non-static variable 'vbad3'}}

int vbad4;
int vbad4 = 10; // expected-warning{{no previous extern declaration for non-static variable 'vbad4'}}
extern int vbad4;

extern int vgood1;
int vgood1;
int vgood1 = 10;
