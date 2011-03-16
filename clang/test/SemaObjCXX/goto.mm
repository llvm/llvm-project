// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR9495
struct NonPOD { NonPOD(); ~NonPOD(); };

@interface A
@end

@implementation A
- (void)method:(bool)b {
  NonPOD np;
  if (b) {
    goto undeclared; // expected-error{{use of undeclared label 'undeclared'}}
  }
}
@end
