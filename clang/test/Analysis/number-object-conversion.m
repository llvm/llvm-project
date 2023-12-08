// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -fblocks -w -Wno-int-conversion -analyzer-checker=osx.NumberObjectConversion %s -verify
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -fblocks -w -Wno-int-conversion -analyzer-checker=osx.NumberObjectConversion -analyzer-config osx.NumberObjectConversion:Pedantic=true -DPEDANTIC %s -verify
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -fblocks -fobjc-arc -w -Wno-int-conversion -analyzer-checker=osx.NumberObjectConversion %s -verify
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -fblocks -fobjc-arc -w -Wno-int-conversion -analyzer-checker=osx.NumberObjectConversion -analyzer-config osx.NumberObjectConversion:Pedantic=true -DPEDANTIC %s -verify

#include "Inputs/system-header-simulator-objc.h"

void takes_boolean(BOOL);
void takes_integer(int);

void bad(NSNumber *p) {
#ifdef PEDANTIC
  if (p) {} // expected-warning{{Converting a pointer value of type 'NSNumber *' to a primitive boolean value; instead, either compare the pointer to nil or call -boolValue}}
  if (!p) {} // expected-warning{{Converting a pointer value of type 'NSNumber *' to a primitive boolean value; instead, either compare the pointer to nil or call -boolValue}}
  (!p) ? 1 : 2; // expected-warning{{Converting a pointer value of type 'NSNumber *' to a primitive boolean value; instead, either compare the pointer to nil or call -boolValue}}
  if (p == 0) {} // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a scalar integer value; instead, either compare the pointer to nil or compare the result of calling a method on 'NSNumber *' to get the scalar value}}
#else
  if (p) {} // no-warning
  if (!p) {} // no-warning
  (!p) ? 1 : 2; // no-warning
  if (p == 0) {} // no-warning
#endif
  (BOOL)p; // expected-warning{{Converting a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to call -boolValue?}}
  if (p > 0) {} // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a scalar integer value; did you mean to compare the result of calling a method on 'NSNumber *' to get the scalar value?}}
  if (p == YES) {} // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to compare the result of calling -boolValue?}}
  if (p == NO) {} // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to compare the result of calling -boolValue?}}
  BOOL x = p; // expected-warning{{Converting a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to call -boolValue?}}
  x = p; // expected-warning{{Converting a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to call -boolValue?}}
  x = (p == YES); // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to compare the result of calling -boolValue?}}
  if (p == 1) {} // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a scalar integer value; did you mean to compare the result of calling a method on 'NSNumber *' to get the scalar value?}}
  int y = p; // expected-warning{{Converting a pointer value of type 'NSNumber *' to a scalar integer value; did you mean to call a method on 'NSNumber *' to get the scalar value?}}
  y = p; // expected-warning{{Converting a pointer value of type 'NSNumber *' to a scalar integer value; did you mean to call a method on 'NSNumber *' to get the scalar value?}}
  takes_boolean(p); // expected-warning{{Converting a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to call -boolValue?}}
  takes_integer(p); // expected-warning{{Converting a pointer value of type 'NSNumber *' to a scalar integer value; did you mean to call a method on 'NSNumber *' to get the scalar value?}}
  takes_boolean(x); // no-warning
  takes_integer(y); // no-warning
}

typedef NSNumber *SugaredNumber;
void bad_sugared(SugaredNumber p) {
  p == YES; // expected-warning{{Comparing a pointer value of type 'SugaredNumber' to a primitive BOOL value; did you mean to compare the result of calling -boolValue?}}
}

@interface I : NSObject {
@public
  NSNumber *ivar;
  NSNumber *prop;
}
- (NSNumber *)foo;
@property(copy) NSNumber *prop;
@end

@implementation I
@synthesize prop;
@end

void bad_ivar(I *i) {
  i->ivar == YES; // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to compare the result of calling -boolValue?}}
  i->prop == YES; // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to compare the result of calling -boolValue?}}
  [i foo] == YES; // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a primitive BOOL value; did you mean to compare the result of calling -boolValue?}}
}

void good(NSNumber *p) {
  if ([p boolValue] == NO) {} // no-warning
  if ([p boolValue] == YES) {} // no-warning
  BOOL x = [p boolValue]; // no-warning
}

void suppression(NSNumber *p) {
  if (p == NULL) {} // no-warning
  if (p == nil) {} // no-warning
}

// Conversion of a pointer to an intptr_t is fine.
typedef long intptr_t;
typedef unsigned long uintptr_t;
typedef long fintptr_t; // Fake, for testing the regex.
void test_intptr_t(NSNumber *p) {
  (long)p; // expected-warning{{Converting a pointer value of type 'NSNumber *' to a scalar integer value; did you mean to call a method on 'NSNumber *' to get the scalar value?}}
  (intptr_t)p; // no-warning
  (unsigned long)p; // expected-warning{{Converting a pointer value of type 'NSNumber *' to a scalar integer value; did you mean to call a method on 'NSNumber *' to get the scalar value?}}
  (uintptr_t)p; // no-warning
  (fintptr_t)p; // expected-warning{{Converting a pointer value of type 'NSNumber *' to a scalar integer value; did you mean to call a method on 'NSNumber *' to get the scalar value?}}
}

// Test macro suppressions.
#define FOO 0
#define BAR 1
void test_macro(NSNumber *p) {
  if (p != BAR) {} // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a scalar integer value; did you mean to compare the result of calling a method on 'NSNumber *' to get the scalar value?}}
#ifdef PEDANTIC
  if (p != FOO) {} // expected-warning{{Comparing a pointer value of type 'NSNumber *' to a scalar integer value; instead, either compare the pointer to nil or compare the result of calling a method on 'NSNumber *' to get the scalar value}}
#else
  if (p != FOO) {} // no-warning
#endif
}

#define NULL_INSIDE_MACRO NULL
void test_NULL_inside_macro(NSNumber *p) {
#ifdef PEDANTIC
  if (p == NULL_INSIDE_MACRO) {} // no-warning
#else
  if (p == NULL_INSIDE_MACRO) {} // no-warning
#endif
}

// Test a different definition of NULL.
#undef NULL
#define NULL 0
void test_non_pointer_NULL(NSNumber *p) {
  if (p == NULL) {} // no-warning
}
