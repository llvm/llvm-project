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
  
  union FooUnion {
    SomeObj* a;
    CFMutableArrayRef b;
    // expected-warning@-1{{Member variable 'b' in 'members::FooUnion' is a retainable type 'CFMutableArrayRef'}}
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

namespace ptr_to_ptr_to_retained {

  struct List {
    CFMutableArrayRef* elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_retained::List' contains a retainable type 'CFMutableArrayRef'}}
  };

  template <typename T, typename S>
  struct TemplateList {
    S* elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_retained::TemplateList<SomeObj, __CFArray *>' contains a raw pointer to retainable type '__CFArray'}}
  };
  TemplateList<SomeObj, CFMutableArrayRef> list;

  struct SafeList {
    RetainPtr<SomeObj>* elements1;
    RetainPtr<CFMutableArrayRef>* elements2;
  };

} // namespace ptr_to_ptr_to_retained
