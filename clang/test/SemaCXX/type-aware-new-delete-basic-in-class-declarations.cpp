// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TADD -std=c++17    -fexperimental-cxx-type-aware-allocators
// RUN: %clang_cc1 -fsyntax-only -verify %s           -std=c++17    -fexperimental-cxx-type-aware-allocators -fcxx-type-aware-destroying-delete
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TAA  -std=c++17 -fno-experimental-cxx-type-aware-allocators

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;

// Basic valid declarations
struct S {
  void *operator new(std::type_identity<S>, size_t);
  void operator delete(std::type_identity<S>, void *);
#if defined(NO_TAA)
  //expected-error@-3 {{type aware allocation operators are disabled}}
  //expected-error@-3 {{type aware allocation operators are disabled}}
#endif
  void operator delete(S *, std::destroying_delete_t);
};

template <typename T> struct S2 {
  void *operator new(std::type_identity<S2<T>>, size_t);
  void operator delete(std::type_identity<S2<T>>, void *);
#if defined(NO_TAA)
  //expected-error@-3 {{type aware allocation operators are disabled}}
  //expected-error@-3 {{type aware allocation operators are disabled}}
#endif
  void operator delete(S2 *, std::destroying_delete_t);
};

struct S3 {
  template <typename T> void *operator new(std::type_identity<T>, size_t);
  template <typename T> void operator delete(std::type_identity<T>, void *);
#if defined(NO_TAA)
  //expected-error@-3 {{type aware allocation operators are disabled}}
  //expected-error@-3 {{type aware allocation operators are disabled}}
#endif
  void operator delete(S3 *, std::destroying_delete_t);
};

struct S4 {
  template <typename T> void *operator new(std::type_identity<T>, size_t);
  template <typename T> void operator delete(std::type_identity<T>, void *);
  template <typename T> void operator delete(std::type_identity<T>, S4 *, std::destroying_delete_t); // #1
#if defined(NO_TAA)
  //expected-error@-4 {{type aware allocation operators are disabled}}
  //expected-error@-4 {{type aware allocation operators are disabled}}
  //expected-error@-4 {{type aware allocation operators are disabled}}
#elif defined(NO_TADD)
  // expected-error@#1 {{type aware destroying delete is not permitted}}
#endif
};

struct S5 {
  template <typename T> void operator delete(std::type_identity<T>, T *); // #2
#if defined(NO_TAA)
  // expected-error@#2 {{type aware allocation operators are disabled}}
#else
  // expected-error@#2 {{'operator delete' cannot take a dependent type as first parameter; use 'void *'}}
#endif
};
