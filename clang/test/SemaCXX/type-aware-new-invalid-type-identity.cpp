// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++26 -DINVALID_TYPE_IDENTITY_VERSION=0
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++26 -DINVALID_TYPE_IDENTITY_VERSION=1
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++26 -DINVALID_TYPE_IDENTITY_VERSION=2
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++26 -DINVALID_TYPE_IDENTITY_VERSION=3
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++26 -DINVALID_TYPE_IDENTITY_VERSION=4
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++26 

namespace std {
#if !defined(INVALID_TYPE_IDENTITY_VERSION)
  // expected-no-diagnostics
  template <class T> struct type_identity {
  };
#elif INVALID_TYPE_IDENTITY_VERSION==0
  struct type_identity {};
  // expected-error@-1 {{std::type_identity must be a class template with a single type parameter}}
#elif INVALID_TYPE_IDENTITY_VERSION==1
  template <class A, class B> struct type_identity {};
  // expected-error@-1 {{std::type_identity must be a class template with a single type parameter}}
#elif INVALID_TYPE_IDENTITY_VERSION==2
  enum type_identity {};
  // expected-error@-1 {{std::type_identity must be a class template with a single type parameter}}
#elif INVALID_TYPE_IDENTITY_VERSION==3
  template <class T> using type_identity = int;
  // expected-error@-1 {{std::type_identity must be a class template with a single type parameter}}
#elif INVALID_TYPE_IDENTITY_VERSION==4
  template <class T> struct inner {};
  template <class T> using type_identity = inner<T>;
  // expected-error@-1 {{std::type_identity must be a class template with a single type parameter}}
#endif
}

using size_t = __SIZE_TYPE__;

struct TestType {};

void f() {
  TestType *t = new TestType;
  delete t;
}
