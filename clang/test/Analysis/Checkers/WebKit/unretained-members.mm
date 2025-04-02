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

namespace ignore_unions {
  union Foo {
    SomeObj* a;
    RetainPtr<SomeObj> b;
    CFMutableArrayRef c;
  };

  template<class T>
  union RefPtr {
    T* a;
  };

  void forceTmplToInstantiate(RefPtr<SomeObj>) {}
}

@interface AnotherObject : NSObject {
  NSString *ns_string;
  // expected-warning@-1{{Instance variable 'ns_string' in 'AnotherObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
  CFStringRef cf_string;
  // expected-warning@-1{{Instance variable 'cf_string' in 'AnotherObject' is a retainable type 'CFStringRef'; member variables must be a RetainPtr}}
}
@property(nonatomic, strong) NSString *prop_string;
// expected-warning@-1{{Property 'prop_string' in 'AnotherObject' is a raw pointer to retainable type 'NSString'; member variables must be a RetainPtr}}
@end
