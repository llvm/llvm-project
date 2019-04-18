// RUN: %clang_analyze_cc1 -triple i686-pc-linux-gnu %s -verify \
// RUN:   -analyzer-checker=security.insecureAPI \
// RUN:   -analyzer-checker=security.FloatLoopCounter

// expected-no-diagnostics

// This file complements 'security-syntax-checks.m', but tests that we omit
// specific checks on platforms where they don't make sense.

// Omit the 'rand' check since 'arc4random' is not available on Linux.
int      rand(void);
double   drand48(void);
double   erand48(unsigned short[3]);
long     jrand48(unsigned short[3]);
void     lcong48(unsigned short[7]);
long     lrand48(void);
long     mrand48(void);
long     nrand48(unsigned short[3]);
long     random(void);
int      rand_r(unsigned *);

void test_rand()
{
  unsigned short a[7];
  unsigned b;
  
  rand();	// no-warning
  drand48();	// no-warning
  erand48(a);	// no-warning
  jrand48(a);	// no-warning
  lcong48(a);	// no-warning
  lrand48();	// no-warning
  mrand48();	// no-warning
  nrand48(a);	// no-warning
  rand_r(&b);	// no-warning
  random();	// no-warning
}
