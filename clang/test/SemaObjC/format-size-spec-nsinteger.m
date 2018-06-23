// RUN: %clang_cc1 -triple thumbv7-apple-ios -Wno-objc-root-class -fsyntax-only -verify -Wformat %s
// RUN: %clang_cc1 -triple thumbv7-apple-ios -Wno-objc-root-class -fsyntax-only -verify -Wformat-pedantic -DPEDANTIC %s

#if !defined(PEDANTIC)
// expected-no-diagnostics
#endif

#if __LP64__
typedef unsigned long NSUInteger;
typedef long NSInteger;
#else
typedef unsigned int NSUInteger;
typedef int NSInteger;
#endif

@class NSString;

extern void NSLog(NSString *format, ...);

void testSizeSpecifier() {
  NSInteger i = 0;
  NSUInteger j = 0;
  NSLog(@"max NSInteger = %zi", i);
  NSLog(@"max NSUinteger = %zu", j);

#if defined(PEDANTIC)
  // expected-warning@-4 {{values of type 'NSInteger' should not be used as format arguments; add an explicit cast to 'long' instead}}
  // expected-warning@-4 {{values of type 'NSUInteger' should not be used as format arguments; add an explicit cast to 'unsigned long' instead}}
#endif
}
