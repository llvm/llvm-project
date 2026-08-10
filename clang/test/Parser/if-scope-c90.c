// RUN: %clang_cc1 -fsyntax-only -verify -std=c90 %s
// expected-no-diagnostics

int f (int z)
{ 
  if (z > (int) sizeof (enum {a, b}))
      return a;
   return b;
} 
