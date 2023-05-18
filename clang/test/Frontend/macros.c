// RUN: %clang_cc1 -DA= -DB=1 -verify -fsyntax-only %s
// expected-no-diagnostics

int a[(B A) == 1 ? 1 : -1];


// PR13747 - Don't warn about unused results with statement exprs in macros.
void stuff(int,int,int);
#define memset(x,y,z) ({ stuff(x,y,z); x; })

void foo(int a, int b, int c) {
  memset(a,b,c);  // No warning!
}
