// RUN: %clang_cc1 -triple x86_64-linux-pc -DIS64 -fsyntax-only -verify -std=c++17 %s 
// RUN: %clang_cc1 -triple x86_64-windows-pc -DIS64 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -triple i386-linux-pc -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -triple i386-windows-pc -DW32 -fsyntax-only -verify -std=c++17 %s
// expected-no-diagnostics

struct Base {};
struct A : virtual Base {
  virtual void n() {}
};

auto p = &A::n;
static_assert(__has_unique_object_representations(decltype(p)));

struct B {
  decltype(p) x;
  int b;
#ifdef IS64
  // required on 64 bit to fill out the tail padding.
  int c;
#endif
};
static_assert(__has_unique_object_representations(B));

struct C { // has padding on Win32, but nothing else.
  decltype(p) x;
};
#ifdef W32
static_assert(!__has_unique_object_representations(C));
#else
static_assert(__has_unique_object_representations(C));
#endif
