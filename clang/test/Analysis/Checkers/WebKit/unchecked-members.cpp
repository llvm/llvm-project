// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUncheckedPtrMemberChecker -verify %s

#include "mock-types.h"

namespace members {

  struct Foo {
  private:
    CheckedObj* a = nullptr;
// expected-warning@-1{{Member variable 'a' in 'members::Foo' is a raw pointer to CheckedPtr capable type 'CheckedObj'}}
    CheckedObj& b;
// expected-warning@-1{{Member variable 'b' in 'members::Foo' is a reference to CheckedPtr capable type 'CheckedObj'}}

    [[clang::suppress]]
    CheckedObj* a_suppressed = nullptr;

    [[clang::suppress]]
    CheckedObj& b_suppressed;

    CheckedPtr<CheckedObj> c;
    CheckedRef<CheckedObj> d;

  public:
    Foo();
  };

  template <typename S>
  struct FooTmpl {
    S* e;
// expected-warning@-1{{Member variable 'e' in 'members::FooTmpl<CheckedObj>' is a raw pointer to CheckedPtr capable type 'CheckedObj'}}
  };

  void forceTmplToInstantiate(FooTmpl<CheckedObj>) { }

} // namespace members

namespace ignore_unions {

  union Foo {
    CheckedObj* a;
    CheckedPtr<CheckedObj> c;
    CheckedRef<CheckedObj> d;
  };

  template<class T>
  union FooTmpl {
    T* a;
  };

  void forceTmplToInstantiate(FooTmpl<CheckedObj>) { }

} // namespace ignore_unions
