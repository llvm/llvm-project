// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUnretainedMemberChecker -fobjc-runtime-has-weak -fobjc-weak -fobjc-arc -verify %s
// expected-no-diagnostics

#include "objc-mock-types.h"

struct Foo {
  __weak NSString *weakPtr = nullptr;
  Foo();
  ~Foo();
  void bar();
};

@interface ObjectWithWeakProperty : NSObject
@property(nonatomic, weak) NSString *weak_prop;
@end

@implementation ObjectWithWeakProperty
@end

NS_REQUIRES_PROPERTY_DEFINITIONS
@interface NoSynthesisObjectWithWeakProperty : NSObject
@property(nonatomic, readonly, weak) NSString *weak_prop;
@end

@implementation NoSynthesisObjectWithWeakProperty
- (NSString *)weak_prop {
  return nil;
}
@end
