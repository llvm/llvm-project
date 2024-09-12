// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#import "mock-types.h"
#import "mock-system-header.h"
#import "../../Inputs/system-header-simulator-for-objc-dealloc.h"

@interface Foo : NSObject

@property (nonatomic, readonly) RefPtr<RefCountable> countable;

- (void)execute;
- (RefPtr<RefCountable>)_protectedRefCountable;
@end

@implementation Foo

- (void)execute {
  self._protectedRefCountable->method();
}

- (RefPtr<RefCountable>)_protectedRefCountable {
  return _countable;
}

@end
