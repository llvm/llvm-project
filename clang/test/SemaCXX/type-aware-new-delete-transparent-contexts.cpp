// RUN: %clang_cc1  -fsyntax-only -verify %s -std=c++26 -fexceptions -DTRANSPARENT_DECL=0
// RUN: %clang_cc1  -fsyntax-only -verify %s -std=c++26 -fexceptions -DTRANSPARENT_DECL=1
// RUN: %clang_cc1  -fsyntax-only -verify %s -std=c++26 -fexceptions -DTRANSPARENT_DECL=2

// expected-no-diagnostics
#if TRANSPARENT_DECL==2
export module Testing;
#endif

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
