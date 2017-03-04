// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -w -analyzer-checker=osx.NumberObjectConversion %s -verify
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -w -analyzer-checker=osx.NumberObjectConversion -analyzer-config osx.NumberObjectConversion:Pedantic=true -DPEDANTIC %s -verify

#define NULL ((void *)0)

typedef const struct __CFNumber *CFNumberRef;

void takes_int(int);

void bad(CFNumberRef p) {
#ifdef PEDANTIC
  if (p) {} // expected-warning{{Converting a pointer value of type 'CFNumberRef' to a primitive boolean value; instead, either compare the pointer to NULL or call CFNumberGetValue()}}
  if (!p) {} // expected-warning{{Converting a pointer value of type 'CFNumberRef' to a primitive boolean value; instead, either compare the pointer to NULL or call CFNumberGetValue()}}
  p ? 1 : 2; // expected-warning{{Converting a pointer value of type 'CFNumberRef' to a primitive boolean value; instead, either compare the pointer to NULL or call CFNumberGetValue()}}
  if (p == 0) {} // expected-warning{{Comparing a pointer value of type 'CFNumberRef' to a primitive integer value; instead, either compare the pointer to NULL or compare the result of calling CFNumberGetValue()}}
#else
  if (p) {} // no-warning
  if (!p) {} // no-warning
  p ? 1 : 2; // no-warning
  if (p == 0) {} // no-warning
#endif
  if (p > 0) {} // expected-warning{{Comparing a pointer value of type 'CFNumberRef' to a primitive integer value; did you mean to compare the result of calling CFNumberGetValue()?}}
  int x = p; // expected-warning{{Converting a pointer value of type 'CFNumberRef' to a primitive integer value; did you mean to call CFNumberGetValue()?}}
  x = p; // expected-warning{{Converting a pointer value of type 'CFNumberRef' to a primitive integer value; did you mean to call CFNumberGetValue()?}}
  takes_int(p); // expected-warning{{Converting a pointer value of type 'CFNumberRef' to a primitive integer value; did you mean to call CFNumberGetValue()?}}
  takes_int(x); // no-warning
}

// Conversion of a pointer to an intptr_t is fine.
typedef long intptr_t;
typedef unsigned long uintptr_t;
typedef long fintptr_t; // Fake, for testing the regex.
void test_intptr_t(CFNumberRef p) {
  (long)p; // expected-warning{{Converting a pointer value of type 'CFNumberRef' to a primitive integer value; did you mean to call CFNumberGetValue()?}}
  (intptr_t)p; // no-warning
  (unsigned long)p; // expected-warning{{Converting a pointer value of type 'CFNumberRef' to a primitive integer value; did you mean to call CFNumberGetValue()?}}
  (uintptr_t)p; // no-warning
  (fintptr_t)p; // expected-warning{{Converting a pointer value of type 'CFNumberRef' to a primitive integer value; did you mean to call CFNumberGetValue()?}}
}

