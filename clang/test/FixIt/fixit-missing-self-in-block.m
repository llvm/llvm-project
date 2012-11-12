// RUN: cp %s %t
// RUN: %clang_cc1 -x objective-c -fobjc-arc -fblocks -fixit %t
// RUN: %clang_cc1 -x objective-c -fobjc-arc -fblocks -Werror %t
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
           _bar = 3;
       }();
  }
@end
