// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1  -fsyntax-only -verify %t/testing.cpp         -std=c++26 -Wno-ext-cxx-type-aware-allocators -fexceptions -DTRANSPARENT_DECL=0
// RUN: %clang_cc1  -fsyntax-only -verify %t/testing.cpp         -std=c++26 -Wno-ext-cxx-type-aware-allocators -fexceptions -DTRANSPARENT_DECL=1
// RUN: %clang_cc1  -fsyntax-only -verify %t/module_testing.cppm -std=c++26 -Wno-ext-cxx-type-aware-allocators -fexceptions -DTRANSPARENT_DECL=2

//--- module_testing.cppm
// expected-no-diagnostics
export module Testing;

#include "testing.inc"

//--- testing.cpp
// expected-no-diagnostics
#include "testing.inc"

//--- testing.inc
namespace std {
  template <class T> struct type_identity {};
  using size_t = __SIZE_TYPE__;
  enum class align_val_t : size_t {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

#if TRANSPARENT_DECL==0
#define BEGIN_TRANSPARENT_DECL extern "C" {
#define END_TRANSPARENT_DECL }
#elif TRANSPARENT_DECL==1
#define BEGIN_TRANSPARENT_DECL extern "C++" {
#define END_TRANSPARENT_DECL }
#elif TRANSPARENT_DECL==2
#define BEGIN_TRANSPARENT_DECL export {
#define END_TRANSPARENT_DECL }
#else
#error unexpected decl kind
#endif

BEGIN_TRANSPARENT_DECL
  void *operator new(std::type_identity<int>, std::size_t, std::align_val_t);
  void operator delete[](std::type_identity<int>, void*, std::size_t, std::align_val_t);
END_TRANSPARENT_DECL

void *operator new[](std::type_identity<int>, std::size_t, std::align_val_t);
void operator delete(std::type_identity<int>, void*, std::size_t, std::align_val_t);

void foo() {
  int *iptr = new int;
  delete iptr;
  int *iarray = new int[5];
  delete [] iarray;
}
