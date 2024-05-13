// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -fsyntax-only -fobjc-arc -x objective-c -fobjc-runtime-has-weak %s.result
// RUN: arcmt-test --args -triple x86_64-apple-macosx10.7 -fblocks -fsyntax-only %s > %t
// RUN: diff %t %s.result

#include "Common.h"

@interface Foo : NSObject
-(Foo *)something;
@end

void bar(void (^block)(void));

void test1(Foo *p) {
  __block Foo *x = p; // __block used just to break cycle.
  bar(^{
    [x something];
  });
}

void test2(Foo *p) {
  __block Foo *x; // __block used as output variable.
  bar(^{
    x = [p something];
  });
}

void test3(Foo *p) {
  __block Foo *x; // __block used as output variable.
  bar(^{
    [x something];
  });
  bar(^{
    x = 0;
  });
}

void test4(Foo *p) {
  __block Foo *x = p; // __block used just to break cycle.
  bar(^{
    [x something];
  });
  bar(^{
    [x something];
  });
}
