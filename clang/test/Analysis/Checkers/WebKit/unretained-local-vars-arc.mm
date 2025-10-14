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

void baz() {
  auto queue = dispatch_queue_create("some queue", nullptr);
  dispatch_queue_get_label(queue);
}

} // namespace raw_ptr

namespace const_global {

extern NSString * const SomeConstant;
extern CFDictionaryRef const SomeDictionary;
extern dispatch_queue_t const SomeDispatch;
void doWork(NSString *, CFDictionaryRef, dispatch_queue_t);
void use_const_global() {
  doWork(SomeConstant, SomeDictionary, SomeDispatch);
}

NSString *provide_str();
CFDictionaryRef provide_dict();
dispatch_queue_t provide_dispatch();
void use_const_local() {
  NSString * const str = provide_str();
  CFDictionaryRef dict = provide_dict();
  // expected-warning@-1{{Local variable 'dict' is unretained and unsafe [alpha.webkit.UnretainedLocalVarsChecker]}}
  auto dispatch = provide_dispatch();
  doWork(str, dict, dispatch);
}

} // namespace const_global

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

- (void)baz {
  auto queue = dispatch_queue_create("some queue", nullptr);
  dispatch_queue_get_label(queue);
}
@end
