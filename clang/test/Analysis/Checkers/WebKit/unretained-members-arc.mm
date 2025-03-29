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
  };

  template<class T, class S>
  struct FooTmpl {
    T* x;
    S y;
// expected-warning@-1{{Member variable 'y' in 'members::FooTmpl<SomeObj, __CFArray *>' is a raw pointer to retainable type}}
  };

  void forceTmplToInstantiate(FooTmpl<SomeObj, CFMutableArrayRef>) {}

  struct [[clang::suppress]] FooSuppressed {
  private:
    SomeObj* a = nullptr;
  };

}
