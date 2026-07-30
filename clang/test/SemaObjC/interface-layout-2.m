// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics
@interface A
{
  int ivar;
}
@end

@interface B : A
- (int)ivar;
@end

@implementation B
- (int)ivar {
  return ivar;
} 
@end
