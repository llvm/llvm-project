// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s

// RUN: %clang_cc1 -std=c++11 -emit-pch -fpch-instantiate-templates -o %t %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s

// expected-no-diagnostics

// Before the patch, this test triggered an assert violation in
// ASTContext::getSubstTemplateTypeParmType.

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

template <typename T>
using Id = T;

template <typename X>
struct Class1 {
  template <typename Y, typename = decltype(Y())>
  struct Nested1;
};

template <typename A>
struct Class2 {
  template <typename B, typename = Id<decltype(B())>>
  struct Nested2;
};

#else

Class2<char> test;

#endif
