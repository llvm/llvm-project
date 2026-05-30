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

    RetainPtr<SomeObj>* g = nullptr;
// expected-warning@-1{{Member variable 'g' in 'members::Foo' is a raw pointer to 'WTF::RetainPtrArc<SomeObj>'}}
    RetainPtr<SomeObj>** h = nullptr;
// expected-warning@-1{{Member variable 'h' in 'members::Foo' contains a raw pointer to 'WTF::RetainPtrArc<SomeObj>'}}
    RetainPtr<SomeObj>* [[clang::annotate_type("webkit.unsafeptr")]] i = nullptr;
    RetainPtr<SomeObj>** [[clang::annotate_type("webkit.unsafeptr")]] j = nullptr;
// expected-warning@-1{{Member variable 'j' in 'members::Foo' contains a raw pointer to 'WTF::RetainPtrArc<SomeObj>'}}
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
    RetainPtr<T>* t;
// expected-warning@-1{{Member variable 't' in 'members::FooTmpl<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' is a raw pointer to 'WTF::RetainPtrArc<SomeObj>'}}
    S* u;
// expected-warning@-1{{Member variable 'u' in 'members::FooTmpl<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' contains a raw pointer to retainable type}}
    RetainPtr<S>* v;
// expected-warning@-1{{Member variable 'v' in 'members::FooTmpl<SomeObj, __CFArray *, NSObject<OS_dispatch_queue> *>' is a raw pointer to 'WTF::RetainPtrArc<__CFArray *>'}}
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

  struct FormerlySafeList {
    RetainPtr<SomeObj>* elements1;
    // expected-warning@-1{{Member variable 'elements1' in 'ptr_to_ptr_to_retained::FormerlySafeList' is a raw pointer to 'WTF::RetainPtrArc<SomeObj>'}}
    RetainPtr<CFMutableArrayRef>* elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_retained::FormerlySafeList' is a raw pointer to 'WTF::RetainPtrArc<__CFArray *>'}}
    OSObjectPtr<dispatch_queue_t> elements3;
  };

  struct Container {
    RetainPtr<SomeObj>* [[clang::annotate_type("webkit.unsafeptr")]] elements1;
    RetainPtr<CFMutableArrayRef>** [[clang::annotate_type("webkit.unsafeptr")]] elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_retained::Container' contains a raw pointer to 'WTF::RetainPtrArc<__CFArray *>'}}
    RetainPtr<SomeObj>* [[clang::annotate_type("webkit.unsafeptr")]]* [[clang::annotate_type("webkit.unsafeptr")]] elements3;
  };

} // namespace ptr_to_ptr_to_retained

@interface AnotherObject : NSObject {
  NSString *ns_string;
  CFStringRef cf_string;
  // expected-warning@-1{{Instance variable 'cf_string' in 'AnotherObject' is a retainable type 'CFStringRef'; member variables must be a RetainPtr}}
  dispatch_queue_t queue;
  RetainPtr<SomeObj>* retainptr_ptr;
  // expected-warning@-1{{Instance variable 'retainptr_ptr' in 'AnotherObject' is a raw pointer to 'WTF::RetainPtrArc<SomeObj>'}}
  RetainPtr<SomeObj>** retainptr_ptr_ptr;
  // expected-warning@-1{{Instance variable 'retainptr_ptr_ptr' in 'AnotherObject' contains a raw pointer to 'WTF::RetainPtrArc<SomeObj>'}}
}
@property(nonatomic, strong) NSString *prop_string1;
@property(nonatomic, assign) NSString *prop_string2;
// expected-warning@-1{{Property 'prop_string2' in 'AnotherObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, unsafe_unretained) NSString *prop_string3;
// expected-warning@-1{{Property 'prop_string3' in 'AnotherObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, readonly) NSString *prop_string4;
@property(nonatomic, readonly) NSString *prop_safe;
@property(nonatomic, readonly) RetainPtr<NSString> *prop_retainptr_ptr;
// expected-warning@-1{{Property 'prop_retainptr_ptr' in 'AnotherObject' is a raw pointer to 'WTF::RetainPtrArc<NSString>'}}
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
  RetainPtr<NSString> *ns_string_retainptr_ptr;
  // expected-warning@-1{{Instance variable 'ns_string_retainptr_ptr' in 'NoSynthObject' is a raw pointer to 'WTF::RetainPtrArc<NSString>'}}
  RetainPtr<NSString> **ns_string_retainptr_ptr_ptr;
  // expected-warning@-1{{Instance variable 'ns_string_retainptr_ptr_ptr' in 'NoSynthObject' contains a raw pointer to 'WTF::RetainPtrArc<NSString>'}}
}
@property(nonatomic, readonly, strong) NSString *prop_string1;
@property(nonatomic, readonly, strong) NSString *prop_string2;
@property(nonatomic, assign) NSString *prop_string3;
// expected-warning@-1{{Property 'prop_string3' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, unsafe_unretained) NSString *prop_string4;
// expected-warning@-1{{Property 'prop_string4' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, unsafe_unretained) dispatch_queue_t prop_string5;
// expected-warning@-1{{Property 'prop_string5' in 'NoSynthObject' is a retainable type 'dispatch_queue_t'}}
@property(nonatomic, readonly, strong) dispatch_queue_t dispatch;
@property(nonatomic, readonly) RetainPtr<SomeObj> *prop_retainptr_ptr;
// expected-warning@-1{{Property 'prop_retainptr_ptr' in 'NoSynthObject' is a raw pointer to 'WTF::RetainPtrArc<SomeObj>'}}
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
@synthesize prop_retainptr_ptr;
@end
