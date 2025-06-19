// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUnretainedMemberChecker -verify %s

@class SystemObject;

#include "objc-mock-types.h"
#include "mock-system-header.h"

namespace members {

  struct Foo {
  private:
    SomeObj* a = nullptr;
// expected-warning@-1{{Member variable 'a' in 'members::Foo' is a raw pointer to retainable type}}

    [[clang::suppress]]
    SomeObj* a_suppressed = nullptr;
// No warning.

  protected:
    RetainPtr<SomeObj> b;
// No warning.

  public:
    SomeObj* c = nullptr;
// expected-warning@-1{{Member variable 'c' in 'members::Foo' is a raw pointer to retainable type}}
    RetainPtr<SomeObj> d;

    CFMutableArrayRef e = nullptr;
// expected-warning@-1{{Member variable 'e' in 'members::Foo' is a retainable type 'CFMutableArrayRef'}}
  };

  template<class T, class S>
  struct FooTmpl {
    T* a;
// expected-warning@-1{{Member variable 'a' in 'members::FooTmpl<SomeObj, __CFArray *>' is a raw pointer to retainable type}}
    S b;
// expected-warning@-1{{Member variable 'b' in 'members::FooTmpl<SomeObj, __CFArray *>' is a raw pointer to retainable type}}
  };

  void forceTmplToInstantiate(FooTmpl<SomeObj, CFMutableArrayRef>) {}

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
  };

  template <typename T, typename S>
  struct TemplateList {
    T** elements1;
    // expected-warning@-1{{Member variable 'elements1' in 'ptr_to_ptr_to_retained::TemplateList<SomeObj, __CFArray *>' contains a raw pointer to retainable type 'SomeObj'}}
    S* elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_retained::TemplateList<SomeObj, __CFArray *>' contains a raw pointer to retainable type '__CFArray'}}
  };
  TemplateList<SomeObj, CFMutableArrayRef> list;

  struct SafeList {
    RetainPtr<SomeObj>* elements1;
    RetainPtr<CFMutableArrayRef>* elements2;
  };

} // namespace ptr_to_ptr_to_retained

@interface AnotherObject : NSObject {
  NSString *ns_string;
  // expected-warning@-1{{Instance variable 'ns_string' in 'AnotherObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
  CFStringRef cf_string;
  // expected-warning@-1{{Instance variable 'cf_string' in 'AnotherObject' is a retainable type 'CFStringRef'; member variables must be a RetainPtr}}
}
@property(nonatomic, strong) NSString *prop_string;
// expected-warning@-1{{Property 'prop_string' in 'AnotherObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@end

NS_REQUIRES_PROPERTY_DEFINITIONS
@interface NoSynthObject : NSObject {
  NSString *ns_string;
  // expected-warning@-1{{Instance variable 'ns_string' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
  CFStringRef cf_string;
  // expected-warning@-1{{Instance variable 'cf_string' in 'NoSynthObject' is a retainable type 'CFStringRef'; member variables must be a RetainPtr}}
}
@property(nonatomic, readonly, strong) NSString *prop_string1;
@property(nonatomic, readonly, strong) NSString *prop_string2;
// expected-warning@-1{{Property 'prop_string2' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'}}
@property(nonatomic, assign) NSString *prop_string3;
// expected-warning@-1{{Property 'prop_string3' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@property(nonatomic, unsafe_unretained) NSString *prop_string4;
// expected-warning@-1{{Property 'prop_string4' in 'NoSynthObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@end

@implementation NoSynthObject
- (NSString *)prop_string1 {
  return nil;
}
@synthesize prop_string2;
@synthesize prop_string3;
@synthesize prop_string4;
@end
