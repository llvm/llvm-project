// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.NoUncountedMemberChecker -verify %s

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
  RefCountable* _uncounted1;
// expected-warning@-1{{Instance variable '_uncounted1' in 'SomeObjC' is a raw pointer to ref-countable type 'RefCountable'}}  
  RefPtr<RefCountable> _counted1;
  [[clang::suppress]] RefCountable* _uncounted2;
}
- (void)doWork;
@end

@implementation SomeObjC {
  RefCountable* _uncounted3;
// expected-warning@-1{{Instance variable '_uncounted3' in 'SomeObjC' is a raw pointer to ref-countable type 'RefCountable'}}
  RefPtr<RefCountable> _counted2;
  [[clang::suppress]] RefCountable* _uncounted4;
}

- (void)doWork {
  doSomeWork();
}

@end
