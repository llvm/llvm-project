// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedCallArgsChecker -fobjc-arc -verify %s

#import "objc-mock-types.h"

SomeObj *provide();
CFMutableArrayRef provide_cf();
void someFunction();
CGImageRef provideImage();
NSString *stringForImage(CGImageRef);

namespace raw_ptr {

void foo() {
  [provide() doWork];
  CFArrayAppendValue(provide_cf(), nullptr);
  // expected-warning@-1{{Call argument for parameter 'theArray' is unretained and unsafe [alpha.webkit.UnretainedCallArgsChecker]}}
}

} // namespace raw_ptr

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
@end
