// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s -DNO_TADD -std=c++23    -fexperimental-cxx-type-aware-allocators
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s           -std=c++23    -fexperimental-cxx-type-aware-allocators -fexperimental-cxx-type-aware-destroying-delete
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s -DNO_TAA  -std=c++23 -fno-experimental-cxx-type-aware-allocators

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;

// Basic valid declarations
struct S {
  void *operator new(std::type_identity<S>, size_t, std::align_val_t); // #1
  void operator delete(std::type_identity<S>, void *, size_t, std::align_val_t); // #2
#if defined(NO_TAA)
  //expected-error@#1 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#2 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif
  void operator delete(S *, std::destroying_delete_t);
};

template <typename T> struct S2 {
  void *operator new(std::type_identity<S2<T>>, size_t, std::align_val_t); // #3
  void operator delete(std::type_identity<S2<T>>, void *, size_t, std::align_val_t); // #4
#if defined(NO_TAA)
  //expected-error@#3 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#4 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif
  void operator delete(S2 *, std::destroying_delete_t);
};

struct S3 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #5
  template <typename T> void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t); // #6
#if defined(NO_TAA)
  //expected-error@#5 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#6 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif
  void operator delete(S3 *, std::destroying_delete_t);
};

struct S4 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #7
  template <typename T> void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t); // #8
  template <typename T> void operator delete(std::type_identity<T>, S4 *, std::destroying_delete_t, size_t, std::align_val_t); // #9
#if defined(NO_TAA)
  //expected-error@#7 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#8 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
  //expected-error@#9 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#elif defined(NO_TADD)
  // expected-error@#9 {{type aware destroying delete is not permitted, enable with '-fexperimental-cxx-type-aware-destroying-delete'}}
#endif
};

struct S5 {
#if !defined(NO_TAA)
  // expected-error@-2 {{declaration of type aware 'operator delete' in 'S5' must have matching type aware 'operator new'}}
  // expected-note@#10 {{unmatched type aware 'operator delete' declared here}}
#endif
  template <typename T> void operator delete(std::type_identity<T>, T *, size_t, std::align_val_t); // #10
#if defined(NO_TAA)
  // expected-error@#10 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#else
  // expected-error@#10 {{type aware 'operator delete' cannot take a dependent type as its second parameter}}
#endif
};

struct S6 {
  template <typename T> void *operator new(std::type_identity<S6>, T, std::align_val_t); // #11
#if defined(NO_TAA)
  // expected-error@#11 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#else
  // expected-error@#11 {{type aware 'operator new' cannot take a dependent type as its second parameter}}
#endif
  template <typename T> void operator delete(std::type_identity<S6>, T, size_t, std::align_val_t); // #12
#if defined(NO_TAA)
  // expected-error@#12 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#else
  // expected-error@#12 {{type aware 'operator delete' cannot take a dependent type as its second parameter}}
#endif
};

#if !defined(NO_TAA)
template <typename U>
struct S7 {
  template <typename T> void *operator new(std::type_identity<T>, U, std::align_val_t); // #13
  // expected-error@#13 {{type aware 'operator new' cannot take a dependent type as its second parameter;}}
  template <typename T> void operator delete(std::type_identity<T>, U, size_t, std::align_val_t); // #14
  // expected-error@#14 {{type aware 'operator delete' cannot take a dependent type as its second parameter;}}
#if !defined(NO_TADD)
  template <typename T> void operator delete(std::type_identity<T>, S7 *, std::destroying_delete_t, U, std::align_val_t); // #15
#endif
  void operator delete(S7 *, std::destroying_delete_t, U); // #16
};

void f() {
  S7<int> s;
  // expected-note@-1 {{in instantiation of template class 'S7<int>' requested here}}
#if !defined(NO_TADD)
  // expected-error@#15 {{type aware destroying 'operator delete' cannot take a dependent type as its fourth parameter; use 'unsigned long' instead}}
#endif
  // expected-error@#16 {{destroying operator delete can have only an optional size and optional alignment parameter}}
}

struct S8 {
  template <typename T, typename U> void *operator new(std::type_identity<T>, U, std::align_val_t); // #17
  // expected-error@#17 {{type aware 'operator new' cannot take a dependent type as its second parameter;}}
  template <typename T, typename U> void operator delete(std::type_identity<T>, U, size_t, std::align_val_t); // #18
  // expected-error@#18 {{type aware 'operator delete' cannot take a dependent type as its second parameter;}}
#if !defined(NO_TADD)
  template <typename T, typename U> void operator delete(std::type_identity<T>, S8 *, std::destroying_delete_t, U, std::align_val_t); // #19
  // expected-error@#19 {{type aware destroying 'operator delete' cannot take a dependent type as its fourth parameter; use 'unsigned long' instead}}
#endif
};

template <typename T> using Alias = T;
template <typename T> using TypeIdentityAlias = std::type_identity<T>;
typedef std::type_identity<double> TypedefAlias;
using UsingAlias = std::type_identity<float>;
struct S9 {
  void *operator new(Alias<size_t>, std::align_val_t);
  template <typename T> void *operator new(Alias<std::type_identity<T>>, Alias<size_t>, std::align_val_t);
  void *operator new(Alias<std::type_identity<int>>, size_t, std::align_val_t);
  template <typename T> void operator delete(Alias<std::type_identity<T>>, void *, size_t, std::align_val_t);
  void operator delete(Alias<std::type_identity<int>>, void *, size_t, std::align_val_t);
};
struct S10 {
  template <typename T> void *operator new(TypeIdentityAlias<T>, size_t, std::align_val_t);
  void *operator new(TypeIdentityAlias<int>, size_t, std::align_val_t);
  template <typename T> void operator delete(TypeIdentityAlias<T>, void *, size_t, std::align_val_t);
  void operator delete(TypeIdentityAlias<int>, void *, size_t, std::align_val_t);
};

void test() {
  S9 *s9 = new S9;
  delete s9;
  S10 *s10 = new S10;
  delete s10;
}

struct S11 {
  template <typename T> void *operator new(TypedefAlias, size_t, std::align_val_t);
  void *operator new(TypedefAlias, size_t, std::align_val_t);
  template <typename T> void operator delete(TypedefAlias, void *, size_t, std::align_val_t);
  void operator delete(TypedefAlias, void *, size_t, std::align_val_t);
};
struct S12 {
  template <typename T> void *operator new(UsingAlias, size_t, std::align_val_t);
  void *operator new(UsingAlias, size_t, std::align_val_t);
  template <typename T> void operator delete(UsingAlias, void *, size_t, std::align_val_t);
  void operator delete(UsingAlias, void *, size_t, std::align_val_t);
};

struct S13 {
  void *operator new(std::type_identity<S13>, size_t, std::align_val_t);
  void operator delete(std::type_identity<S13>, void*, size_t, std::align_val_t);
};

#endif
