// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify -fobjc-arc %s

#if __has_feature(objc_arc)
// expected-no-diagnostics
#endif

@interface SomeClass
@end

void simpleStrongPointerValue() {
  SomeClass *x;
  if (x) {}
#if !__has_feature(objc_arc)
// expected-warning@-2{{Branch condition evaluates to a garbage value}}
#endif
}

void simpleArray() {
  SomeClass *vlaArray[5];

  if (vlaArray[0]) {}
#if !__has_feature(objc_arc)
// expected-warning@-2{{Branch condition evaluates to a garbage value}}
#endif
}

void variableLengthArray() {
   int count = 1;
   SomeClass * vlaArray[count];

   if (vlaArray[0]) {}
#if !__has_feature(objc_arc)
  // expected-warning@-2{{Branch condition evaluates to a garbage value}}
#endif
}

void variableLengthArrayWithExplicitStrongAttribute() {
   int count = 1;
   __attribute__((objc_ownership(strong))) SomeClass * vlaArray[count];

   if (vlaArray[0]) {}
#if !__has_feature(objc_arc)
  // expected-warning@-2{{Branch condition evaluates to a garbage value}}
#endif
}
