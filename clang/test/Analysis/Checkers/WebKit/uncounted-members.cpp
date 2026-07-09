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
    Foo();

    RefCountable silenceWarningAboutInit;
    RefCountable& c = silenceWarningAboutInit;
// expected-warning@-1{{Member variable 'c' in 'members::Foo' is a raw reference to ref-countable type 'RefCountable'}}
    Ref<RefCountable> d;
    Ref<RefCountable>* e;
// expected-warning@-1{{Member variable 'e' in 'members::Foo' is a raw pointer to 'Ref<RefCountable>'}}
    Ref<RefCountable>& f;
// expected-warning@-1{{Member variable 'f' in 'members::Foo' is a raw reference to 'Ref<RefCountable>'}}
    Ref<RefCountable>** g;
// expected-warning@-1{{Member variable 'g' in 'members::Foo' contains a raw pointer to 'Ref<RefCountable>'}}
    RefPtr<RefCountable>* h;
// expected-warning@-1{{Member variable 'h' in 'members::Foo' is a raw pointer to 'RefPtr<RefCountable>'}}
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

namespace unions {
  union Foo {
    RefCountable* a;
    // expected-warning@-1{{Member variable 'a' in 'unions::Foo' is a raw pointer to ref-countable type 'RefCountable'}}
    RefPtr<RefCountable> b;
    Ref<RefCountable> c;
    RefCountable** d;
    // expected-warning@-1{{Member variable 'd' in 'unions::Foo' contains a raw pointer to ref-countable type 'RefCountable'}}
    Ref<RefCountable>* e;
    // expected-warning@-1{{Member variable 'e' in 'unions::Foo' is a raw pointer to 'Ref<RefCountable>'}}
  };

  template<class T>
  union FooTmpl {
    T* a;
    // expected-warning@-1{{Member variable 'a' in 'unions::FooTmpl<RefCountable>' is a raw pointer to ref-countable type 'RefCountable'}}
  };

  void forceTmplToInstantiate(FooTmpl<RefCountable>) {}
} // unions

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

namespace ptr_to_ptr_to_ref_counted {

  struct List {
    RefCountable** elements;
    // expected-warning@-1{{Member variable 'elements' in 'ptr_to_ptr_to_ref_counted::List' contains a raw pointer to ref-countable type 'RefCountable'}}
  };

  template <typename T>
  struct TemplateList {
    T** elements;
    // expected-warning@-1{{Member variable 'elements' in 'ptr_to_ptr_to_ref_counted::TemplateList<RefCountable>' contains a raw pointer to ref-countable type 'RefCountable'}}
  };
  TemplateList<RefCountable> list;

  struct FormerlySafeList {
    RefPtr<RefCountable>* elements;
    // expected-warning@-1{{Member variable 'elements' in 'ptr_to_ptr_to_ref_counted::FormerlySafeList' is a raw pointer to 'RefPtr<RefCountable>'}}
  };

  struct Container {
    RefPtr<CheckedObj>* [[clang::annotate_type("webkit.unsafeptr")]] elements1;
    RefPtr<CheckedObj>** [[clang::annotate_type("webkit.unsafeptr")]] elements2;
    // expected-warning@-1{{Member variable 'elements2' in 'ptr_to_ptr_to_ref_counted::Container' contains a raw pointer to 'RefPtr<CheckedObj>'}}
    Ref<CheckedObj>* [[clang::annotate_type("webkit.unsafeptr")]]* [[clang::annotate_type("webkit.unsafeptr")]] elements3;
  };

} // namespace ptr_to_ptr_to_ref_counted
