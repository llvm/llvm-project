// Test that infinite loop was fixed.

// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class -include %s -include %s %s

// With PCH
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s -chain-include %s -chain-include %s

#ifndef HEADER1
#define HEADER1
//===----------------------------------------------------------------------===//
// Primary header

@class I;

@interface I2
@property (readonly) id prop1;
@end

//===----------------------------------------------------------------------===//
#elif !defined(HEADER2)
#define HEADER2
#if !defined(HEADER1)
#error Header inclusion order messed up
#endif

//===----------------------------------------------------------------------===//
// Dependent header

@interface I
@end

@interface I(Cat1)
@end

@interface I(Cat2)
@end

@interface I2()
@property (readwrite,assign) id prop1;
@property (copy) id prop2;
@end

//===----------------------------------------------------------------------===//
#else
//===----------------------------------------------------------------------===//

// expected-note@30 {{receiver is instance of class declared here}}
void f(I* i) {
  [i meth]; // expected-warning {{not found}}
}

@implementation I2
@synthesize prop1, prop2;
@end

//===----------------------------------------------------------------------===//
#endif
