// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.WebKitNoUncountedMemberChecker -verify %s

#include "mock-types.h"

namespace members {
  struct Foo {
  private:
    RefCountable* a = nullptr;
// expected-warning@-1{{Member variable 'a' in 'members::Foo' is a raw pointer to ref-countable type 'RefCountable'}}

  protected:
    RefPtr<RefCountable> b;

  public:
    RefCountable silenceWarningAboutInit;
    RefCountable& c = silenceWarningAboutInit;
// expected-warning@-1{{Member variable 'c' in 'members::Foo' is a reference to ref-countable type 'RefCountable'}}
    Ref<RefCountable> d;
  };

  template<class T>
  struct FooTmpl {
    T* a;
// expected-warning@-1{{Member variable 'a' in 'members::FooTmpl<RefCountable>' is a raw pointer to ref-countable type 'RefCountable'}}
  };

  void forceTmplToInstantiate(FooTmpl<RefCountable>) {}
}

namespace ignore_unions {
  union Foo {
    RefCountable* a;
    RefPtr<RefCountable> b;
    Ref<RefCountable> c;
  };

  template<class T>
  union RefPtr {
    T* a;
  };

  void forceTmplToInstantiate(RefPtr<RefCountable>) {}
}
