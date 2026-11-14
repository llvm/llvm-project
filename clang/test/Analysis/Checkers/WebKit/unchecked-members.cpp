// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUncheckedPtrMemberChecker -verify %s

#include "mock-types.h"

namespace members {

  struct Foo {
  private:
    CheckedObj* a = nullptr;
// expected-warning@-1{{Member variable 'a' (of 'members::Foo') is a raw pointer to CheckedPtr-capable type 'CheckedObj'}}
    CheckedObj& b;
// expected-warning@-1{{Member variable 'b' (of 'members::Foo') is a reference to CheckedPtr-capable type 'CheckedObj'}}

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
  };

  void forceTmplToInstantiate(FooTmpl<CheckedObj>) { }
  // expected-warning@-1{{Member variable 'e' (of 'members::FooTmpl<CheckedObj>') is a raw pointer to CheckedPtr-capable type 'CheckedObj'}}

} // namespace members

namespace unions {

  union Foo {
    CheckedObj* a;
    // expected-warning@-1{{Member variable 'a' (of 'unions::Foo') is a raw pointer to CheckedPtr-capable type 'CheckedObj'}}
    CheckedPtr<CheckedObj> c;
    CheckedRef<CheckedObj> d;
  };

  template<class T>
  union FooTmpl {
    T* a;
  };

  void forceTmplToInstantiate(FooTmpl<CheckedObj>) { }
  // expected-warning@-1{{Member variable 'a' (of 'unions::FooTmpl<CheckedObj>') is a raw pointer to CheckedPtr-capable type 'CheckedObj'}}

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
    // expected-warning@-1{{Member variable 'elements' (of 'ptr_to_ptr_to_checked_ptr_capable::List') contains a raw pointer to CheckedPtr-capable type 'CheckedObj'}}
  };

  template <typename T>
  struct TemplateList {
    T** elements;
  };
  TemplateList<CheckedObj> list;
  // expected-warning@-1{{Member variable 'elements' (of 'ptr_to_ptr_to_checked_ptr_capable::TemplateList<CheckedObj>') contains a raw pointer to CheckedPtr-capable type 'CheckedObj'}}

  struct SafeList {
    CheckedPtr<CheckedObj>* elements;
  };

} // namespace ptr_to_ptr_to_checked_ptr_capable
