// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu

struct a { // expected-error {{structure 'a' is too large, which exceeds maximum allowed size of 1152921504606846976 bytes}}
  char x[1ull<<60]; 
  char x2[1ull<<60]; 
};

a z[1];
long long x() { return sizeof(a); }
long long x2() { return sizeof(a::x); }
long long x3() { return sizeof(a::x2); }
long long x4() { return sizeof(z); }

