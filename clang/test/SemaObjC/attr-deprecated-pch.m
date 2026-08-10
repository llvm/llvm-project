// RUN: %clang_cc1 -fsyntax-only -DBOTH -verify %s
// If the decls come from a pch, the behavior shouldn't change:
// RUN: %clang_cc1 -x objective-c-header %s -emit-pch -o %t
// RUN: %clang_cc1 -DUSES -include-pch %t -fsyntax-only -verify %s
// expected-no-diagnostics

// The slightly strange ifdefs are so that the command that builds the gch file
// doesn't need any -D switches, for these would get embedded in the gch.

#ifndef USES
@interface Interface1
- (void)partiallyUnavailableMethod;
@end
@interface Interface2
- (void)partiallyUnavailableMethod __attribute__((unavailable));
@end
#endif

#if defined(USES) || defined(BOTH)
void f(id a) {
  [a partiallyUnavailableMethod];  // no warning, `a` could be an Interface1.
}
#endif
