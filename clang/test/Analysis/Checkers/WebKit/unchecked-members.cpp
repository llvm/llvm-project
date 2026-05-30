// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUncheckedPtrMemberChecker -verify %s

#include "mock-types.h"

namespace members {

  struct Foo {
  private:
    CheckedObj* a = nullptr;
// expected-warning@-1{{Member variable 'a' in 'members::Foo' is a raw pointer to CheckedPtr capable type 'CheckedObj'}}
    CheckedObj& b;
// expected-warning@-1{{Member variable 'b' in 'members::Foo' is a raw reference to CheckedPtr capable type 'CheckedObj'}}

    [[clang::suppress]]
    CheckedObj* a_suppressed = nullptr;

    [[clang::suppress]]
    CheckedObj& b_suppressed;

    CheckedPtr<CheckedObj> c;
    CheckedRef<CheckedObj> d;
    CheckedRef<RefCountable>* e;
// expected-warning@-1{{Member variable 'e' in 'members::Foo' is a raw pointer to 'CheckedRef<RefCountable>'}}
    CheckedRef<RefCountable>& f;
// expected-warning@-1{{Member variable 'f' in 'members::Foo' is a raw reference to 'CheckedRef<RefCountable>'}}
    CheckedRef<RefCountable>** g;
// expected-warning@-1{{Member variable 'g' in 'members::Foo' contains a raw pointer to 'CheckedRef<RefCountable>'}}
    CheckedRef<RefCountable>* h;
// expected-warning@-1{{Member variable 'h' in 'members::Foo' is a raw pointer to 'CheckedRef<RefCountable>'}}

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

namespace unions {

  union Foo {
    CheckedObj* a;
    // expected-warning@-1{{Member variable 'a' in 'unions::Foo' is a raw pointer to CheckedPtr capable type 'CheckedObj'}}
    CheckedPtr<CheckedObj> b;
    CheckedRef<CheckedObj> c;
    CheckedObj** d;
    // expected-warning@-1{{Member variable 'd' in 'unions::Foo' contains a raw pointer to CheckedPtr capable type 'CheckedObj'}}
    CheckedPtr<CheckedObj>* e;
    // expected-warning@-1{{Member variable 'e' in 'unions::Foo' is a raw pointer to 'CheckedPtr<CheckedObj>'}}
  };

  template<class T>
  union FooTmpl {
    T* a;
    // expected-warning@-1{{Member variable 'a' in 'unions::FooTmpl<CheckedObj>' is a raw pointer to CheckedPtr capable type 'CheckedObj'}}
  };

  void forceTmplToInstantiate(FooTmpl<CheckedObj>) { }

} // namespace unions

namespace checked_ptr_ref_ptr_capable {

  RefCountableAndCheckable* provide();
  void foo() {
    RefPtr<RefCountableAndCheckable> foo = provide();
  }

} // checked_ptr_ref_ptr_capable

namespace ptr_to_ptr_to_checked_ptr_capable {

  struct List {
    CheckedObj** elements;
    // expected-warning@-1{{Member variable 'elements' in 'ptr_to_ptr_to_checked_ptr_capable::List' contains a raw pointer to CheckedPtr capable type 'CheckedObj'}}
  };

  template <typename T>
  struct TemplateList {
    T** elements;
    // expected-warning@-1{{Member variable 'elements' in 'ptr_to_ptr_to_checked_ptr_capable::TemplateList<CheckedObj>' contains a raw pointer to CheckedPtr capable type 'CheckedObj'}}
  };
  TemplateList<CheckedObj> list;

  struct FormerlySafeList {
    CheckedPtr<CheckedObj>* elements;
    // expected-warning@-1{{Member variable 'elements' in 'ptr_to_ptr_to_checked_ptr_capable::FormerlySafeList' is a raw pointer to 'CheckedPtr<CheckedObj>'}}
  };

  struct Container {
    CheckedPtr<CheckedObj>* [[clang::annotate_type("webkit.unsafeptr")]] elements1;
    CheckedPtr<CheckedObj>** [[clang::annotate_type("webkit.unsafeptr")]] elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_checked_ptr_capable::Container' contains a raw pointer to 'CheckedPtr<CheckedObj>'}}
    CheckedRef<CheckedObj>* [[clang::annotate_type("webkit.unsafeptr")]]* [[clang::annotate_type("webkit.unsafeptr")]] elements3;
  };

} // namespace ptr_to_ptr_to_checked_ptr_capable
