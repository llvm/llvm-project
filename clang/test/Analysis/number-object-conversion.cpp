// RUN: %clang_cc1 -triple i386-apple-darwin10 -w -std=c++11 -analyze -analyzer-checker=osx.NumberObjectConversion %s -verify
// RUN: %clang_cc1 -triple i386-apple-darwin10 -w -std=c++11 -analyze -analyzer-checker=osx.NumberObjectConversion -analyzer-config osx.NumberObjectConversion:Pedantic=true -DPEDANTIC %s -verify

#define NULL ((void *)0)
#include "Inputs/system-header-simulator-cxx.h" // for nullptr

class OSBoolean {
public:
  virtual bool isTrue() const;
  virtual bool isFalse() const;
};

extern const OSBoolean *const &kOSBooleanFalse;
extern const OSBoolean *const &kOSBooleanTrue;

void takes_bool(bool);

void bad(const OSBoolean *p) {
#ifdef PEDANTIC
  if (p) {} // expected-warning{{Converting 'const class OSBoolean *' to a plain boolean value for branching; please compare the pointer to NULL or nullptr instead to suppress this warning}}
  if (!p) {} // expected-warning{{Converting 'const class OSBoolean *' to a plain boolean value for branching; please compare the pointer to NULL or nullptr instead to suppress this warning}}
  p ? 1 : 2; // expected-warning{{Converting 'const class OSBoolean *' to a plain boolean value for branching; please compare the pointer to NULL or nullptr instead to suppress this warning}}
  (bool)p; // expected-warning{{Converting 'const class OSBoolean *' to a plain bool value; please compare the pointer to NULL or nullptr instead to suppress this warning}}
#endif
  bool x = p; // expected-warning{{Converting 'const class OSBoolean *' to a plain bool value; pointer value is being used instead}}
  x = p; // expected-warning{{Converting 'const class OSBoolean *' to a plain bool value; pointer value is being used instead}}
  takes_bool(p); // expected-warning{{Converting 'const class OSBoolean *' to a plain bool value; pointer value is being used instead}}
  takes_bool(x); // no-warning
}

typedef bool sugared_bool;
typedef const OSBoolean *sugared_OSBoolean;
void bad_sugared(sugared_OSBoolean p) {
  sugared_bool x = p; // expected-warning{{Converting 'const class OSBoolean *' to a plain bool value; pointer value is being used instead}}
}

void good(const OSBoolean *p) {
  bool x = p->isTrue(); // no-warning
  (bool)p->isFalse(); // no-warning
  if (p == kOSBooleanTrue) {} // no-warning
}

void suppression(const OSBoolean *p) {
  if (p == NULL) {} // no-warning
  bool y = (p == nullptr); // no-warning
}

// Conversion of a pointer to an intptr_t is fine.
typedef long intptr_t;
typedef unsigned long uintptr_t;
typedef long fintptr_t; // Fake, for testing the regex.
void test_intptr_t(const OSBoolean *p) {
  (long)p; // expected-warning{{Converting 'const class OSBoolean *' to a plain integer value; pointer value is being used instead}}
  (intptr_t)p; // no-warning
  (unsigned long)p; // expected-warning{{Converting 'const class OSBoolean *' to a plain integer value; pointer value is being used instead}}
  (uintptr_t)p; // no-warning
  (fintptr_t)p; // expected-warning{{Converting 'const class OSBoolean *' to a plain integer value; pointer value is being used instead}}
}

// Test a different definition of NULL.
#undef NULL
#define NULL 0
void test_non_pointer_NULL(const OSBoolean *p) {
  if (p == NULL) {} // no-warning
}
