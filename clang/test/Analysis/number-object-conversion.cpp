// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -w -std=c++11 -analyzer-checker=osx.NumberObjectConversion %s -verify
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -w -std=c++11 -analyzer-checker=osx.NumberObjectConversion -analyzer-config osx.NumberObjectConversion:Pedantic=true -DPEDANTIC %s -verify

#define NULL ((void *)0)
#include "Inputs/system-header-simulator-cxx.h" // for nullptr

class OSBoolean {
public:
  virtual bool isTrue() const;
  virtual bool isFalse() const;
};

class OSNumber {
public:
  virtual bool isEqualTo(const OSNumber *);
  virtual unsigned char unsigned8BitValue() const;
  virtual unsigned short unsigned16BitValue() const;
  virtual unsigned int unsigned32BitValue() const;
  virtual unsigned long long unsigned64BitValue() const;
};

extern const OSBoolean *const &kOSBooleanFalse;
extern const OSBoolean *const &kOSBooleanTrue;

void takes_bool(bool);

void bad_boolean(const OSBoolean *p) {
#ifdef PEDANTIC
  if (p) {} // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive boolean value; instead, either compare the pointer to nullptr or call getValue()}}
  if (!p) {} // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive boolean value; instead, either compare the pointer to nullptr or call getValue()}}
  p ? 1 : 2; // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive boolean value; instead, either compare the pointer to nullptr or call getValue()}}
#else
  if (p) {} // no-warning
  if (!p) {} // no-warning
  p ? 1 : 2; // no-warning
#endif
  (bool)p; // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive bool value; did you mean to call getValue()?}}
  bool x = p; // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive bool value; did you mean to call getValue()?}}
  x = p; // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive bool value; did you mean to call getValue()?}}
  takes_bool(p); // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive bool value; did you mean to call getValue()?}}
  takes_bool(x); // no-warning
}

void bad_number(const OSNumber *p) {
#ifdef PEDANTIC
  if (p) {} // expected-warning{{Converting a pointer value of type 'class OSNumber *' to a scalar boolean value; instead, either compare the pointer to nullptr or call a method on 'class OSNumber *' to get the scalar value}}
  if (!p) {} // expected-warning{{Converting a pointer value of type 'class OSNumber *' to a scalar boolean value; instead, either compare the pointer to nullptr or call a method on 'class OSNumber *' to get the scalar value}}
  p ? 1 : 2; // expected-warning{{Converting a pointer value of type 'class OSNumber *' to a scalar boolean value; instead, either compare the pointer to nullptr or call a method on 'class OSNumber *' to get the scalar value}}
  if (p == 0) {} // expected-warning{{Comparing a pointer value of type 'class OSNumber *' to a scalar integer value; instead, either compare the pointer to nullptr or compare the result of calling a method on 'class OSNumber *' to get the scalar value}}
#else
  if (p) {} // no-warning
  if (!p) {} // no-warning
  p ? 1 : 2; // no-warning
  if (p == 0) {} // no-warning
#endif
  (int)p; // expected-warning{{Converting a pointer value of type 'class OSNumber *' to a scalar integer value; did you mean to call a method on 'class OSNumber *' to get the scalar value?}}
  takes_bool(p); // expected-warning{{Converting a pointer value of type 'class OSNumber *' to a scalar bool value; did you mean to call a method on 'class OSNumber *' to get the scalar value?}}
}

typedef bool sugared_bool;
typedef const OSBoolean *sugared_OSBoolean;
void bad_sugared(sugared_OSBoolean p) {
  sugared_bool x = p; // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive bool value; did you mean to call getValue()?}}
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
  (long)p; // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive integer value; did you mean to call getValue()?}}
  (intptr_t)p; // no-warning
  (unsigned long)p; // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive integer value; did you mean to call getValue()?}}
  (uintptr_t)p; // no-warning
  (fintptr_t)p; // expected-warning{{Converting a pointer value of type 'class OSBoolean *' to a primitive integer value; did you mean to call getValue()?}}
}

// Test a different definition of NULL.
#undef NULL
#define NULL 0
void test_non_pointer_NULL(const OSBoolean *p) {
  if (p == NULL) {} // no-warning
}
