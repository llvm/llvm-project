// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin10 -fobjc-arc %s -include %s -include %s

// With PCH
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin10 -fobjc-arc %s -chain-include %s -chain-include %s

// expected-no-diagnostics

#ifndef HEADER1
#define HEADER1
//===----------------------------------------------------------------------===//
// Primary header

@interface I
+(void)meth;
@end

//===----------------------------------------------------------------------===//
#elif !defined(HEADER2)
#define HEADER2
#if !defined(HEADER1)
#error Header inclusion order messed up
#endif

//===----------------------------------------------------------------------===//
// Dependent header

@interface I()
@property (assign) id prop;
+(void)meth2;
@end

//===----------------------------------------------------------------------===//
#else
//===----------------------------------------------------------------------===//

void foo(I *i) {
  [I meth];
  [I meth2];
  i.prop = 0;
}

//===----------------------------------------------------------------------===//
#endif
