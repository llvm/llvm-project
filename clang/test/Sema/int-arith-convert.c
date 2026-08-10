// RUN: %clang_cc1 -triple=i686-linux-gnu -fsyntax-only -verify %s
// expected-no-diagnostics

// Check types are the same through redeclaration
unsigned long x;
__typeof(1u+1l) x;

unsigned y;
__typeof(1+1u) y;
__typeof(1u+1) y;

long long z;
__typeof(1ll+1u) z;
