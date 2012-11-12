// RUN: %clang_cc1 -x objective-c -fobjc-arc -fblocks -Wimplicit-retain-self -verify %s
// rdar://11194874

@interface Root @end

@interface I : Root
{
  int _bar;
}
@end

@implementation I
  - (void)foo{
      ^{
           _bar = 3; // expected-warning {{block implicitly retains 'self'; explicitly mention 'self' to indicate this is intended behavior}}
       }();
  }
@end
