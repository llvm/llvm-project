// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUnretainedMemberChecker -fobjc-arc -verify %s

#include "objc-mock-types.h"

namespace members {

  struct Foo {
  private:
    SomeObj* a = nullptr;

    [[clang::suppress]]
    SomeObj* a_suppressed = nullptr;

  protected:
    RetainPtr<SomeObj> b;

  public:
    SomeObj* c = nullptr;
    RetainPtr<SomeObj> d;

    CFMutableArrayRef e = nullptr;
// expected-warning@-1{{Member variable 'e' in 'members::Foo' is a retainable type 'CFMutableArrayRef'}}

    dispatch_queue_t f = nullptr;
  };
  
  union FooUnion {
    SomeObj* a;
    CFMutableArrayRef b;
    // expected-warning@-1{{Member variable 'b' in 'members::FooUnion' is a retainable type 'CFMutableArrayRef'}}
    dispatch_queue_t c;
  };

  template<class T, class S, class R>
  struct FooTmpl {
    T* x;
    S y;
// expected-warning@-1{{Member variable 'y' in 'members::FooTmpl<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' is a raw pointer to retainable type}}
    R z;
  };

  void forceTmplToInstantiate(FooTmpl<SomeObj, CFMutableArrayRef, dispatch_queue_t>) {}

  struct [[clang::suppress]] FooSuppressed {
  private:
    SomeObj* a = nullptr;
  };

}

namespace ptr_to_ptr_to_retained {

  struct List {
    CFMutableArrayRef* elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_retained::List' contains a retainable type 'CFMutableArrayRef'}}
  };

  template <typename T, typename S, typename R>
  struct TemplateList {
    T* elements1;
    S* elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_retained::TemplateList<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' contains a raw pointer to retainable type '__CFArray'}}
    R* elements3;
  };
  TemplateList<SomeObj, CFMutableArrayRef, dispatch_queue_t> list;

  struct SafeList {
    RetainPtr<SomeObj>* elements1;
    RetainPtr<CFMutableArrayRef>* elements2;
    OSObjectPtr<dispatch_queue_t> elements3;
  };

} // namespace ptr_to_ptr_to_retained

@interface AnotherObject : NSObject {
  NSString *ns_string;
  CFStringRef cf_string;
  // expected-warning@-1{{Instance variable 'cf_string' in 'AnotherObject' is a retainable type 'CFStringRef'; member variables must be a RetainPtr}}
  dispatch_queue_t queue;
}
@property(nonatomic, strong) NSString *prop_string1;
@property(nonatomic, assign) NSString *prop_string2;
// expected-warning@-1{{Property 'prop_string2' in 'AnotherObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, unsafe_unretained) NSString *prop_string3;
// expected-warning@-1{{Property 'prop_string3' in 'AnotherObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, readonly) NSString *prop_string4;
@property(nonatomic, readonly) NSString *prop_safe;
@end

@implementation AnotherObject
- (NSString *)prop_safe {
  return nil;
}
@end

// No warnings for @interface declaration itself. 
@interface InterfaceOnlyObject : NSObject
@property(nonatomic, strong) NSString *prop_string1;
@property(nonatomic, assign) NSString *prop_string2;
@property(nonatomic, unsafe_unretained) NSString *prop_string3;
@property(nonatomic, readonly) NSString *prop_string4;
@property(nonatomic, readonly) dispatch_queue_t prop_string5;
@end

NS_REQUIRES_PROPERTY_DEFINITIONS
@interface NoSynthObject : NSObject {
  NSString *ns_string;
  CFStringRef cf_string;
  // expected-warning@-1{{Instance variable 'cf_string' in 'NoSynthObject' is a retainable type 'CFStringRef'; member variables must be a RetainPtr}}
}
@property(nonatomic, readonly, strong) NSString *prop_string1;
@property(nonatomic, readonly, strong) NSString *prop_string2;
@property(nonatomic, assign) NSString *prop_string3;
// expected-warning@-1{{Property 'prop_string3' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, unsafe_unretained) NSString *prop_string4;
// expected-warning@-1{{Property 'prop_string4' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, unsafe_unretained) dispatch_queue_t prop_string5;
// expected-warning@-1{{Property 'prop_string5' in 'NoSynthObject' is a retainable type 'dispatch_queue_t'}}
@end

@implementation NoSynthObject
- (NSString *)prop_string1 {
  return nil;
}
@synthesize prop_string2;
@synthesize prop_string3;
@synthesize prop_string4;
@synthesize prop_string5;
@end
