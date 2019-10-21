// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,nullability -verify %s

// expected-no-diagnostics

@class C;

#pragma clang assume_nonnull begin
@interface I
- foo:(C *)c;
@end
#pragma clang assume_nonnull end

@interface J
@property C *c;
@end

J *conjure_J();

@implementation I
- (void)bar {
  if (self) { // no-crash
    J *j = conjure_J();
    if (j.c)
      [self bar];
    // FIXME: Should warn.
    [self foo:j.c]; // no-warning
  }
}
@end

@implementation J
@end
