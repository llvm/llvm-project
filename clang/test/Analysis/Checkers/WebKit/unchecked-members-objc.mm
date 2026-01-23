// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUncheckedPtrMemberChecker -verify %s

#include "mock-types.h"

__attribute__((objc_root_class))
@interface NSObject
+ (instancetype) alloc;
- (instancetype) init;
- (instancetype)retain;
- (void)release;
@end

void doSomeWork();

@interface SomeObjC : NSObject {
  CheckedObj* _unchecked1;
// expected-warning@-1{{Instance variable '_unchecked1' in 'SomeObjC' is a raw pointer to CheckedPtr capable type 'CheckedObj'}}  
  CheckedPtr<CheckedObj> _counted1;
  [[clang::suppress]] CheckedObj* _unchecked2;
}
- (void)doWork;
@end

@implementation SomeObjC {
  CheckedObj* _unchecked3;
// expected-warning@-1{{Instance variable '_unchecked3' in 'SomeObjC' is a raw pointer to CheckedPtr capable type 'CheckedObj'}}
  CheckedPtr<CheckedObj> _counted2;
  [[clang::suppress]] CheckedObj* _unchecked4;
}

- (void)doWork {
  doSomeWork();
}

@end
