// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#import "mock-types.h"
#import "mock-system-header.h"
#import "../../Inputs/system-header-simulator-for-objc-dealloc.h"

@interface Foo : NSObject {
  const Ref<RefCountable> _obj1;
  const RefPtr<RefCountable> _obj2;
  Ref<RefCountable> _obj3;
}

@property (nonatomic, readonly) RefPtr<RefCountable> countable;

- (void)execute;
- (RefPtr<RefCountable>)_protectedRefCountable;
@end

@implementation Foo

- (void)execute {
  self._protectedRefCountable->method();
  _obj1->method();
  _obj1.get().method();
  (*_obj2).method();
  _obj3->method();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
}

- (RefPtr<RefCountable>)_protectedRefCountable {
  return _countable;
}

@end

class RefCountedObject {
public:
  void ref() const;
  void deref() const;
  Ref<RefCountedObject> copy() const;
  void method();
};

@interface WrapperObj : NSObject

- (Ref<RefCountedObject>)_protectedWebExtensionControllerConfiguration;

@end

static void foo(WrapperObj *configuration) {
  configuration._protectedWebExtensionControllerConfiguration->copy();
}
