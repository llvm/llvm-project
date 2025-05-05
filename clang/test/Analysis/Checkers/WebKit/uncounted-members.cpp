// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.NoUncountedMemberChecker -verify %s

#include "mock-types.h"
#include "mock-system-header.h"

namespace members {
  struct Foo {
  private:
    RefCountable* a = nullptr;
// expected-warning@-1{{Member variable 'a' in 'members::Foo' is a raw pointer to ref-countable type 'RefCountable'}}

    [[clang::suppress]]
    RefCountable* a_suppressed = nullptr;

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

  struct [[clang::suppress]] FooSuppressed {
  private:
    RefCountable* a = nullptr;
  };
} // members

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
} // ignore_unions

namespace ignore_system_header {

void foo(RefCountable* t) {
  MemberVariable<RefCountable> var { t };
  var.obj->method();
}

} // ignore_system_header

namespace ignore_non_ref_countable {
  struct Foo {
  };

  struct Bar {
    Foo* foo;
  };
} // ignore_non_ref_countable

namespace checked_ptr_ref_ptr_capable {

  RefCountableAndCheckable* provide();
  void foo() {
    CheckedPtr<RefCountableAndCheckable> foo = provide();
  }

} // checked_ptr_ref_ptr_capable
