// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedCallArgsChecker -fobjc-arc -verify %s

#import "objc-mock-types.h"

SomeObj *provide();
CFMutableArrayRef provide_cf();
dispatch_queue_t provide_os();
void someFunction();
CGImageRef provideImage();
NSString *stringForImage(CGImageRef);

namespace raw_ptr {

void foo() {
  [provide() doWork];
  CFArrayAppendValue(provide_cf(), nullptr);
  // expected-warning@-1{{Call argument for parameter 'theArray' is unretained and unsafe [alpha.webkit.UnretainedCallArgsChecker]}}
  dispatch_queue_get_label(provide_os());
}

} // namespace raw_ptr

namespace const_global {

extern NSString * const SomeConstant;
extern CFDictionaryRef const SomeDictionary;
extern dispatch_queue_t const SomeDispatch;
void doWork(NSString *str, CFDictionaryRef dict, dispatch_queue_t dispatch);
void use_const_global() {
  doWork(SomeConstant, SomeDictionary, SomeDispatch);
}

NSString *provide_str();
CFDictionaryRef provide_dict();
dispatch_queue_t provide_dispatch();
void use_const_local() {
  doWork(provide_str(), provide_dict(), provide_dispatch());
  // expected-warning@-1{{Call argument for parameter 'dict' is unretained and unsafe}}
}

} // namespace const_global

@interface AnotherObj : NSObject
- (void)foo:(SomeObj *)obj;
- (SomeObj *)getSomeObj;
@end

@implementation AnotherObj
- (void)foo:(SomeObj*)obj {
  [obj doWork];
  [provide() doWork];
  CFArrayAppendValue(provide_cf(), nullptr);
  // expected-warning@-1{{Call argument for parameter 'theArray' is unretained and unsafe [alpha.webkit.UnretainedCallArgsChecker]}}
}

- (SomeObj *)getSomeObj {
    return provide();
}

- (void)doWorkOnSomeObj {
    [[self getSomeObj] doWork];
}

- (CGImageRef)createImage {
  return provideImage();
}

- (NSString *)convertImage {
  RetainPtr<CGImageRef> image = [self createImage];
  return stringForImage(image.get());
}

- (const char *)dispatchLabel {
  OSObjectPtr obj = provide_os();
  return dispatch_queue_get_label(obj.get());
}
@end
