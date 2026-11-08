// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.NoUncountedMemberChecker -verify %s

#include "mock-types.h"
#include "mock-system-header.h"

namespace members {
  struct Foo {
  private:
    RefCountable* a = nullptr;
// expected-warning@-1{{Member variable 'a' (of 'members::Foo') is a raw pointer to RefPtr-capable type 'RefCountable'}}

    [[clang::suppress]]
    RefCountable* a_suppressed = nullptr;

  protected:
    RefPtr<RefCountable> b;

  public:
    RefCountable silenceWarningAboutInit;
    RefCountable& c = silenceWarningAboutInit;
// expected-warning@-1{{Member variable 'c' (of 'members::Foo') is a reference to RefPtr-capable type 'RefCountable'}}
    Ref<RefCountable> d;
  };

  template<class T>
  struct FooTmpl {
    T* a;
  };

  void forceTmplToInstantiate(FooTmpl<RefCountable>) {}
  // expected-warning@-1{{Member variable 'a' (of 'members::FooTmpl<RefCountable>') is a raw pointer to RefPtr-capable type 'RefCountable'}}

  struct [[clang::suppress]] FooSuppressed {
  private:
    RefCountable* a = nullptr;
  };
} // members

namespace unions {
  union Foo {
    RefCountable* a;
    // expected-warning@-1{{Member variable 'a' (of 'unions::Foo') is a raw pointer to RefPtr-capable type 'RefCountable'}}
    RefPtr<RefCountable> b;
    Ref<RefCountable> c;
  };

  template<class T>
  union FooTmpl {
    T* a;
  };

  void forceTmplToInstantiate(FooTmpl<RefCountable>) {}
  // expected-warning@-1{{Member variable 'a' (of 'unions::FooTmpl<RefCountable>') is a raw pointer to RefPtr-capable type 'RefCountable'}}
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
    // expected-warning@-1{{Member variable 'elements' (of 'ptr_to_ptr_to_ref_counted::List') contains a raw pointer to RefPtr-capable type 'RefCountable'}}
  };

  template <typename T>
  struct TemplateList {
    T** elements;
  };
  TemplateList<RefCountable> list;
  // expected-warning@-1{{Member variable 'elements' (of 'ptr_to_ptr_to_ref_counted::TemplateList<RefCountable>') contains a raw pointer to RefPtr-capable type 'RefCountable'}}

  struct SafeList {
    RefPtr<RefCountable>* elements;
  };

} // namespace ptr_to_ptr_to_ref_counted

namespace template_instantiation {

  template <typename T>
  struct Holder {
    T* dependent;
    RefCountable* nonDependent;
    // expected-warning@-1{{Member variable 'nonDependent' (of 'template_instantiation::Holder') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  };

  Holder<int> intHolder;
  Holder<RefCountable> refCountableHolder;
  // expected-warning@-1{{Member variable 'dependent' (of 'template_instantiation::Holder<RefCountable>') is a raw pointer to RefPtr-capable type 'RefCountable'}}

  struct OtherRefCountable {
    void ref() const;
    void deref() const;
  };
  Holder<OtherRefCountable> otherHolder;
  // expected-warning@-1{{Member variable 'dependent' (of 'template_instantiation::Holder<template_instantiation::OtherRefCountable>') is a raw pointer to RefPtr-capable type 'template_instantiation::OtherRefCountable'}}

  template <typename T>
  struct Outer {
    struct Inner {
      T* ptr;
    };
    Inner inner;
  };
  Outer<RefCountable> outer;
  // expected-warning@-1{{Member variable 'ptr' (of 'template_instantiation::Outer<RefCountable>::Inner') is a raw pointer to RefPtr-capable type 'RefCountable'}}

  template <typename T>
  struct ExplicitlyInstantiated {
    T* ptr;
  };
  template struct ExplicitlyInstantiated<RefCountable>;
  // expected-warning@-1{{Member variable 'ptr' (of 'template_instantiation::ExplicitlyInstantiated<RefCountable>') is a raw pointer to RefPtr-capable type 'RefCountable'}}

  template <typename T>
  struct Specialized {
    T* ptr;
  };
  template <>
  struct Specialized<RefCountable> {
    RefCountable* ptr;
    // expected-warning@-1{{Member variable 'ptr' (of 'template_instantiation::Specialized<RefCountable>') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  };
  Specialized<RefCountable> specialized;

  template <typename T>
  struct Partial {
    T value;
  };
  template <typename T>
  struct Partial<T*> {
    T* ptr;
  };
  Partial<RefCountable*> partial;
  // expected-warning@-1{{Member variable 'ptr' (of 'template_instantiation::Partial<RefCountable *>') is a raw pointer to RefPtr-capable type 'RefCountable'}}

  struct Nested {
    void ref() const;
    void deref() const;
  };

  template <typename T>
  struct Inner {
    T* ptr;
  };

  template <typename T>
  struct Middle {
    Inner<T> inner;
  };

  template <typename T>
  struct Top {
    Middle<T> middle;
  };

  Top<Nested> top;
  // expected-warning@-1{{Member variable 'ptr' (of 'template_instantiation::Inner<template_instantiation::Nested>') is a raw pointer to RefPtr-capable type 'template_instantiation::Nested'}}

  struct UsedInFunction {
    void ref() const;
    void deref() const;
  };

  template <typename T>
  void useInner() {
    Inner<T> inner;
  }

  template <typename T>
  void callUseInner() {
    useInner<T>();
  }

  void callUseInnerWithUsedInFunction() {
    callUseInner<UsedInFunction>();
    // expected-warning@-1{{Member variable 'ptr' (of 'template_instantiation::Inner<template_instantiation::UsedInFunction>') is a raw pointer to RefPtr-capable type 'template_instantiation::UsedInFunction'}}
  }

  struct NonDependentlyUsed {
    void ref() const;
    void deref() const;
  };

  template <typename T>
  struct NonDependent {
    Inner<NonDependentlyUsed> inner;
    // expected-warning@-1{{Member variable 'ptr' (of 'template_instantiation::Inner<template_instantiation::NonDependentlyUsed>') is a raw pointer to RefPtr-capable type 'template_instantiation::NonDependentlyUsed'}}
  };

  NonDependent<int> nonDependentInt;
  NonDependent<char> nonDependentChar;

} // namespace template_instantiation
