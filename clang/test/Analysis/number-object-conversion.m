// RUN: %clang_cc1 -triple i386-apple-darwin10 -fblocks -w -analyze -analyzer-checker=osx.NumberObjectConversion %s -verify
// RUN: %clang_cc1 -triple i386-apple-darwin10 -fblocks -w -analyze -analyzer-checker=osx.NumberObjectConversion -analyzer-config osx.NumberObjectConversion:Pedantic=true -DPEDANTIC %s -verify
// RUN: %clang_cc1 -triple i386-apple-darwin10 -fblocks -fobjc-arc -w -analyze -analyzer-checker=osx.NumberObjectConversion %s -verify
// RUN: %clang_cc1 -triple i386-apple-darwin10 -fblocks -fobjc-arc -w -analyze -analyzer-checker=osx.NumberObjectConversion -analyzer-config osx.NumberObjectConversion:Pedantic=true -DPEDANTIC %s -verify

#include "Inputs/system-header-simulator-objc.h"

void takes_boolean(BOOL);
void takes_integer(int);

void bad(NSNumber *p) {
#ifdef PEDANTIC
  if (p) {} // expected-warning{{Converting 'NSNumber *' to a plain boolean value for branching; please compare the pointer to nil instead to suppress this warning}}
  if (!p) {} // expected-warning{{Converting 'NSNumber *' to a plain boolean value for branching; please compare the pointer to nil instead to suppress this warning}}
  (!p) ? 1 : 2; // expected-warning{{Converting 'NSNumber *' to a plain boolean value for branching; please compare the pointer to nil instead to suppress this warning}}
  (BOOL)p; // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; please compare the pointer to nil instead to suppress this warning}}
  if (p == 0) {} // expected-warning{{Converting 'NSNumber *' to a plain integer value; please compare the pointer to nil instead to suppress this warning}}
  if (p > 0) {} // expected-warning{{Converting 'NSNumber *' to a plain integer value; pointer value is being used instead}}
#endif
  if (p == YES) {} // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
  if (p == NO) {} // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
  BOOL x = p; // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
  x = p; // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
  x = (p == YES); // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
  if (p == 1) {} // expected-warning{{Converting 'NSNumber *' to a plain integer value; pointer value is being used instead}}
  int y = p; // expected-warning{{Converting 'NSNumber *' to a plain integer value; pointer value is being used instead}}
  y = p; // expected-warning{{Converting 'NSNumber *' to a plain integer value; pointer value is being used instead}}
  takes_boolean(p); // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
  takes_integer(p); // expected-warning{{Converting 'NSNumber *' to a plain integer value; pointer value is being used instead}}
  takes_boolean(x); // no-warning
  takes_integer(y); // no-warning
}

typedef NSNumber *SugaredNumber;
void bad_sugared(SugaredNumber p) {
  p == YES; // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
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
  i->ivar == YES; // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
  i->prop == YES; // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
  [i foo] == YES; // expected-warning{{Converting 'NSNumber *' to a plain BOOL value; pointer value is being used instead}}
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
  (long)p; // expected-warning{{Converting 'NSNumber *' to a plain integer value; pointer value is being used instead}}
  (intptr_t)p; // no-warning
  (unsigned long)p; // expected-warning{{Converting 'NSNumber *' to a plain integer value; pointer value is being used instead}}
  (uintptr_t)p; // no-warning
  (fintptr_t)p; // expected-warning{{Converting 'NSNumber *' to a plain integer value; pointer value is being used instead}}
}

// Test a different definition of NULL.
#undef NULL
#define NULL 0
void test_non_pointer_NULL(NSNumber *p) {
  if (p == NULL) {} // no-warning
}
