// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedLocalVarsChecker -fobjc-arc -verify %s

#import "objc-mock-types.h"

SomeObj *provide();
void someFunction();

namespace raw_ptr {

void foo() {
  SomeObj *bar = [[SomeObj alloc] init];
  [bar doWork];
}

void foo2() {
  SomeObj *bar = provide();
  [bar doWork];
}

void bar() {
  CFMutableArrayRef array = CFArrayCreateMutable(kCFAllocatorDefault, 10);
  // expected-warning@-1{{Local variable 'array' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  CFArrayAppendValue(array, nullptr);
}

} // namespace raw_ptr

@interface AnotherObj : NSObject
- (void)foo:(SomeObj *)obj;
@end

@implementation AnotherObj
- (void)foo:(SomeObj*)obj {
  SomeObj* obj2 = obj;
  SomeObj* obj3 = provide();
  obj = nullptr;
  [obj2 doWork];
  [obj3 doWork];
}

- (void)bar {
  CFMutableArrayRef array = CFArrayCreateMutable(kCFAllocatorDefault, 10);
  // expected-warning@-1{{Local variable 'array' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  CFArrayAppendValue(array, nullptr);
}
@end
