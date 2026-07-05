// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUnretainedMemberChecker -verify %s

@class SystemObject;

#include "objc-mock-types.h"
#include "mock-system-header.h"

namespace members {

  struct Foo {
  private:
    SomeObj* a = nullptr;
// expected-warning@-1{{Member variable 'a' in 'members::Foo' is a raw pointer to retainable type}}
    dispatch_queue_t a2 = nullptr;
// expected-warning@-1{{Member variable 'a2' in 'members::Foo' is a retainable type 'dispatch_queue_t'}}

    [[clang::suppress]]
    SomeObj* a_suppressed = nullptr;
// No warning.

  protected:
    RetainPtr<SomeObj> b;
// No warning.
    OSObjectPtr<dispatch_queue_t> b2;
// No warning.

  public:
    SomeObj* c = nullptr;
// expected-warning@-1{{Member variable 'c' in 'members::Foo' is a raw pointer to retainable type}}
    RetainPtr<SomeObj> d;
    OSObjectPtr<dispatch_queue_t> d2;

    CFMutableArrayRef e = nullptr;
// expected-warning@-1{{Member variable 'e' in 'members::Foo' is a retainable type 'CFMutableArrayRef'}}

  };

  template<class T, class S, class R>
  struct FooTmpl {
    T* a;
// expected-warning@-1{{Member variable 'a' in 'members::FooTmpl<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' is a raw pointer to retainable type}}
    S b;
// expected-warning@-1{{Member variable 'b' in 'members::FooTmpl<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' is a raw pointer to retainable type}}
    R c;
// expected-warning@-1{{Member variable 'c' in 'members::FooTmpl<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' is a raw pointer to retainable type}}
  };

  void forceTmplToInstantiate(FooTmpl<SomeObj, CFMutableArrayRef, dispatch_queue_t>) {}

  struct [[clang::suppress]] FooSuppressed {
  private:
    SomeObj* a = nullptr;
// No warning.
  };

}

namespace unions {
  union Foo {
    SomeObj* a;
    // expected-warning@-1{{Member variable 'a' in 'unions::Foo' is a raw pointer to retainable type 'SomeObj'}}
    RetainPtr<SomeObj> b;
    CFMutableArrayRef c;
    // expected-warning@-1{{Member variable 'c' in 'unions::Foo' is a retainable type 'CFMutableArrayRef'}}
    dispatch_queue_t d;
    // expected-warning@-1{{Member variable 'd' in 'unions::Foo' is a retainable type 'dispatch_queue_t'}}
  };

  template<class T>
  union FooTempl {
    T* a;
    // expected-warning@-1{{Member variable 'a' in 'unions::FooTempl<SomeObj>' is a raw pointer to retainable type 'SomeObj'}}
  };

  void forceTmplToInstantiate(FooTempl<SomeObj>) {}
}

namespace ptr_to_ptr_to_retained {

  struct List {
    SomeObj** elements1;
    // expected-warning@-1{{Member variable 'elements1' in 'ptr_to_ptr_to_retained::List' contains a raw pointer to retainable type 'SomeObj'}}
    CFMutableArrayRef* elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_retained::List' contains a retainable type 'CFMutableArrayRef'}}
    dispatch_queue_t* elements3;
    // expected-warning@-1{{Member variable 'elements3' in 'ptr_to_ptr_to_retained::List' contains a retainable type 'dispatch_queue_t'}}
  };

  template <typename T, typename S, typename R>
  struct TemplateList {
    T** elements1;
    // expected-warning@-1{{Member variable 'elements1' in 'ptr_to_ptr_to_retained::TemplateList<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' contains a raw pointer to retainable type 'SomeObj'}}
    S* elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_retained::TemplateList<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' contains a raw pointer to retainable type '__CFArray'}}
    R* elements3;
    // expected-warning@-1{{Member variable 'elements3' in 'ptr_to_ptr_to_retained::TemplateList<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' contains a raw pointer to retainable type 'NSObject'}}
  };
  TemplateList<SomeObj, CFMutableArrayRef, dispatch_queue_t> list;

  struct SafeList {
    RetainPtr<SomeObj>* elements1;
    RetainPtr<CFMutableArrayRef>* elements2;
  };

} // namespace ptr_to_ptr_to_retained

@interface AnotherObject : NSObject {
  NSString *ns_string;
  // expected-warning@-1{{Instance variable 'ns_string' in 'AnotherObject' is a raw pointer to retainable type 'NSString'}}
  CFStringRef cf_string;
  // expected-warning@-1{{Instance variable 'cf_string' in 'AnotherObject' is a retainable type 'CFStringRef'}}
  dispatch_queue_t dispatch;
  // expected-warning@-1{{Instance variable 'dispatch' in 'AnotherObject' is a retainable type 'dispatch_queue_t'}}
}
@property(nonatomic, readonly, strong) NSString *prop_string;
@property(nonatomic, readonly) NSString *prop_safe;
@end

@implementation AnotherObject
- (NSString *)prop_safe {
  return nil;
}
@end

@interface DerivedObject : AnotherObject {
  NSNumber *ns_number;
  // expected-warning@-1{{Instance variable 'ns_number' in 'DerivedObject' is a raw pointer to retainable type 'NSNumber'}}
  CGImageRef cg_image;
  // expected-warning@-1{{Instance variable 'cg_image' in 'DerivedObject' is a retainable type 'CGImageRef'}}
  dispatch_queue_t os_dispatch;
  // expected-warning@-1{{Instance variable 'os_dispatch' in 'DerivedObject' is a retainable type 'dispatch_queue_t'}}
}
@property(nonatomic, strong) NSNumber *prop_number;
@property(nonatomic, readonly) NSString *prop_string;
@end

@implementation DerivedObject
- (NSString *)prop_string {
  return nil;
}
@end

// No warnings for @interface declaration itself. 
@interface InterfaceOnlyObject : NSObject
@property(nonatomic, strong) NSString *prop_string1;
@property(nonatomic, assign) NSString *prop_string2;
@property(nonatomic, unsafe_unretained) NSString *prop_string3;
@property(nonatomic, readonly) NSString *prop_string4;
@end

@interface InterfaceOnlyObject2 : NSObject
@property(nonatomic, strong) NSString *prop_string1;
@property(nonatomic, assign) NSString *prop_string2;
@property(nonatomic, unsafe_unretained) NSString *prop_string3;
// expected-warning@-1{{Property 'prop_string3' in 'DerivedObject2' is a raw pointer to retainable type 'NSString'}}
@property(nonatomic, readonly) NSString *prop_string4;
@end

@interface DerivedObject2 : InterfaceOnlyObject2
@property(nonatomic, readonly) NSString *prop_string5;
// expected-warning@-1{{Property 'prop_string5' in 'DerivedObject2' is a raw pointer to retainable type 'NSString'}}
@end

@implementation DerivedObject2
@synthesize prop_string3;
@end

NS_REQUIRES_PROPERTY_DEFINITIONS
@interface NoSynthObject : NSObject {
  NSString *ns_string;
  // expected-warning@-1{{Instance variable 'ns_string' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'}}
  CFStringRef cf_string;
  // expected-warning@-1{{Instance variable 'cf_string' in 'NoSynthObject' is a retainable type 'CFStringRef'}}
  dispatch_queue_t dispatch;
  // expected-warning@-1{{Instance variable 'dispatch' in 'NoSynthObject' is a retainable type 'dispatch_queue_t'}}
}
@property(nonatomic, readonly, strong) NSString *prop_string1;
@property(nonatomic, readonly, strong) NSString *prop_string2;
@property(nonatomic, assign) NSString *prop_string3;
// expected-warning@-1{{Property 'prop_string3' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, unsafe_unretained) NSString *prop_string4;
// expected-warning@-1{{Property 'prop_string4' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, copy) NSString *prop_string5;
@property(nonatomic, readonly, strong) dispatch_queue_t dispatch;
@end

@implementation NoSynthObject
- (NSString *)prop_string1 {
  return nil;
}
@synthesize prop_string2;
@synthesize prop_string3;
@synthesize prop_string4;
@synthesize prop_string5;
- (dispatch_queue_t)dispatch {
  return nil;
}
@end
