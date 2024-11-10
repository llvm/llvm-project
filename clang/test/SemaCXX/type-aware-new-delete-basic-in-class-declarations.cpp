// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TADD -std=c++17    -fexperimental-cxx-type-aware-allocators
// RUN: %clang_cc1 -fsyntax-only -verify %s           -std=c++17    -fexperimental-cxx-type-aware-allocators -fexperimental-cxx-type-aware-destroying-delete
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TAA  -std=c++17 -fno-experimental-cxx-type-aware-allocators

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;

// Basic valid declarations
struct S {
  void *operator new(std::type_identity<S>, size_t); // #1
  void operator delete(std::type_identity<S>, void *); // #2
#if defined(NO_TAA)
  //expected-error@#1 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#2 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif
  void operator delete(S *, std::destroying_delete_t);
};

template <typename T> struct S2 {
  void *operator new(std::type_identity<S2<T>>, size_t); // #3
  void operator delete(std::type_identity<S2<T>>, void *); // #4
#if defined(NO_TAA)
  //expected-error@#3 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#4 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif
  void operator delete(S2 *, std::destroying_delete_t);
};

struct S3 {
  template <typename T> void *operator new(std::type_identity<T>, size_t); // #5
  template <typename T> void operator delete(std::type_identity<T>, void *); // #6
#if defined(NO_TAA)
  //expected-error@#5 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#6 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif
  void operator delete(S3 *, std::destroying_delete_t);
};

struct S4 {
  template <typename T> void *operator new(std::type_identity<T>, size_t); // #7
  template <typename T> void operator delete(std::type_identity<T>, void *); // #8
  template <typename T> void operator delete(std::type_identity<T>, S4 *, std::destroying_delete_t); // #9
#if defined(NO_TAA)
  //expected-error@#7 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#8 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#9 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#elif defined(NO_TADD)
  // expected-error@#9 {{type aware destroying delete is not permitted, enable with '-fexperimental-cxx-type-aware-destroying-delete'}}
#endif
};

struct S5 {
  template <typename T> void operator delete(std::type_identity<T>, T *); // #10
#if defined(NO_TAA)
  // expected-error@#10 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#else
  // expected-error@#10 {{'operator delete' cannot take a dependent type as first parameter; use 'void *'}}
#endif
};
