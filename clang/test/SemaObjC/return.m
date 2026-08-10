// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-noreturn -fobjc-exceptions -Wno-objc-root-class %s

int test1(void) {
  id a;
  @throw a;
}

// PR5286
void test2(int a) {
  while (1) {
    if (a)
      return;
  }
}

// PR5286
void test3(int a) {  // expected-warning {{function 'test3' could be declared with attribute 'noreturn'}}
  while (1) {
    if (a)
      @throw (id)0;
  }
}

// This code always returns, we should not issue a noreturn warning.
@class NSException;
@class NSString;
NSString *rdar_4289832(void) {  // no-warning
    @try
    {
        return @"a";
    }
    @catch(NSException *exception)
    {
        return @"b";
    }
    @finally
    {
    }
}

void exit(int) __attribute__((noreturn));
@interface rdar10098695
@end

@implementation rdar10098695
- (void)method { // expected-warning{{method 'method' could be declared with attribute 'noreturn'}}
  exit(1);
}
@end
