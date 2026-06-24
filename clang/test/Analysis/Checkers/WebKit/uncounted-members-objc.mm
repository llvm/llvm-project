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
  RefCountable** _ptr_to_ptr_to_uncounted;
// expected-warning@-1{{Instance variable '_ptr_to_ptr_to_uncounted' in 'SomeObjC' contains a raw pointer to ref-countable type 'RefCountable'}}  
  RefPtr<RefCountable>* ptr_to_refptr1;
// expected-warning@-1{{Instance variable 'ptr_to_refptr1' in 'SomeObjC' is a raw pointer to 'RefPtr<RefCountable>'}}
  RefPtr<RefCountable>* [[clang::annotate_type("webkit.unsafeptr")]] _ptr_to_refptr2;
  RefPtr<RefCountable>** _ptr_to_refptr3;
// expected-warning@-1{{Instance variable '_ptr_to_refptr3' in 'SomeObjC' contains a raw pointer to 'RefPtr<RefCountable>'}}
  Ref<RefCountable>* [[clang::annotate_type("webkit.unsafeptr")]] _ptr_to_ref1;
  Ref<RefCountable>** _ptr_to_ref2;
// expected-warning@-1{{Instance variable '_ptr_to_ref2' in 'SomeObjC' contains a raw pointer to 'Ref<RefCountable>'}}
}
@property (nonatomic) RefCountable **obj1;
// expected-warning@-1{{Property 'obj1' in 'SomeObjC' contains a raw pointer to ref-countable type 'RefCountable'}}
@property (nonatomic) RefPtr<RefCountable> *obj2;
// expected-warning@-1{{Property 'obj2' in 'SomeObjC' is a raw pointer to 'RefPtr<RefCountable>'}}
@property (nonatomic) RefPtr<RefCountable> **obj3;
// expected-warning@-1{{Property 'obj3' in 'SomeObjC' contains a raw pointer to 'RefPtr<RefCountable>'}}
@property(nonatomic, readonly) Ref<RefCountable> *syn_prop;
// expected-warning@-1{{Property 'syn_prop' in 'SomeObjC' is a raw pointer to 'Ref<RefCountable>'}}
- (void)doWork;
@end

@implementation SomeObjC {
  RefCountable* _uncounted3;
// expected-warning@-1{{Instance variable '_uncounted3' in 'SomeObjC' is a raw pointer to ref-countable type 'RefCountable'}}
  RefPtr<RefCountable> _counted2;
  [[clang::suppress]] RefCountable* _uncounted4;
  RefCountable** _uncounted5;
// expected-warning@-1{{Instance variable '_uncounted5' in 'SomeObjC' contains a raw pointer to ref-countable type 'RefCountable'}}  
  RefPtr<RefCountable>* _ptr_to_refptr1;
// expected-warning@-1{{Instance variable '_ptr_to_refptr1' in 'SomeObjC' is a raw pointer to 'RefPtr<RefCountable>'}}
}

@synthesize syn_prop;

- (void)doWork {
  doSomeWork();
}

@end
