// RUN: %clang_cc1 -triple arm64-apple-macosx -std=c++26 -fsyntax-only -verify=expected,clangext %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -std=c++26 -Wno-ext-cxx-type-aware-allocators -fsyntax-only -verify %s

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
  // clangext-warning@#1 {{type aware allocators are a Clang extension}}
  // clangext-warning@#2 {{type aware allocators are a Clang extension}}
  void operator delete(S *, std::destroying_delete_t);
};

template <typename T> struct S2 {
  void *operator new(std::type_identity<S2<T>>, size_t, std::align_val_t); // #3
  void operator delete(std::type_identity<S2<T>>, void *, size_t, std::align_val_t); // #4
  // clangext-warning@#3 {{type aware allocators are a Clang extension}}
  // clangext-warning@#4 {{type aware allocators are a Clang extension}}
  void operator delete(S2 *, std::destroying_delete_t);
};

struct S3 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #5
  template <typename T> void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t); // #6
  // clangext-warning@#5 {{type aware allocators are a Clang extension}}
  // clangext-warning@#6 {{type aware allocators are a Clang extension}}
  void operator delete(S3 *, std::destroying_delete_t);
};

struct S4 {
  template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #7
  template <typename T> void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t); // #8
  template <typename T> void operator delete(std::type_identity<T>, S4 *, std::destroying_delete_t, size_t, std::align_val_t); // #9
  // clangext-warning@#7 {{type aware allocators are a Clang extension}}
  // clangext-warning@#8 {{type aware allocators are a Clang extension}}
  // expected-error@#9 {{destroying delete is not permitted to be type aware}}
};

struct S5 {
  template <typename T> void operator delete(std::type_identity<T>, T *, size_t, std::align_val_t); // #10
  // expected-error@#10 {{type aware 'operator delete' cannot take a dependent type as its 2nd parameter}}
  // clangext-warning@#10 {{type aware allocators are a Clang extension}}
};

struct S6 {
  template <typename T> void *operator new(std::type_identity<S6>, T, std::align_val_t); // #11
  // expected-error@#11 {{type aware 'operator new' cannot take a dependent type as its 2nd parameter}}
  // clangext-warning@#11 {{type aware allocators are a Clang extension}}
  template <typename T> void operator delete(std::type_identity<S6>, T, size_t, std::align_val_t); // #12
  // expected-error@#12 {{type aware 'operator delete' cannot take a dependent type as its 2nd parameter}}
  // clangext-warning@#12 {{type aware allocators are a Clang extension}}
};

template <typename U>
struct S7 {
  template <typename T> void *operator new(std::type_identity<T>, U, std::align_val_t); // #13
  // expected-error@#13 {{type aware 'operator new' cannot take a dependent type as its 2nd parameter;}}
  // clangext-warning@#13 {{type aware allocators are a Clang extension}}
  template <typename T> void operator delete(std::type_identity<T>, U, size_t, std::align_val_t); // #14
  // expected-error@#14 {{type aware 'operator delete' cannot take a dependent type as its 2nd parameter;}}
  // clangext-warning@#14 {{type aware allocators are a Clang extension}}
  template <typename T> void operator delete(std::type_identity<T>, S7 *, std::destroying_delete_t, U, std::align_val_t); // #15
  // expected-error@#15 {{destroying delete is not permitted to be type aware}}
  void operator delete(S7 *, std::destroying_delete_t, U); // #16
};

void f() {
  S7<int> s;
  // expected-note@-1 {{in instantiation of template class 'S7<int>' requested here}}
  // expected-error@#16 {{destroying operator delete can have only an optional size and optional alignment parameter}}
}

struct S8 {
  template <typename T, typename U> void *operator new(std::type_identity<T>, U, std::align_val_t); // #17
  // expected-error@#17 {{type aware 'operator new' cannot take a dependent type as its 2nd parameter;}}
  // clangext-warning@#17 {{type aware allocators are a Clang extension}}
  template <typename T, typename U> void operator delete(std::type_identity<T>, U, size_t, std::align_val_t); // #18
  // expected-error@#18 {{type aware 'operator delete' cannot take a dependent type as its 2nd parameter;}}
  // clangext-warning@#18 {{type aware allocators are a Clang extension}}
  template <typename T, typename U> void operator delete(std::type_identity<T>, S8 *, std::destroying_delete_t, U, std::align_val_t); // #19
  // expected-error@#19 {{destroying delete is not permitted to be type aware}}
};

template <typename T> using Alias = T;
template <typename T> using TypeIdentityAlias = std::type_identity<T>;
typedef std::type_identity<double> TypedefAlias;
using UsingAlias = std::type_identity<float>;
struct S9 {
  void *operator new(Alias<size_t>, std::align_val_t);
  template <typename T> void *operator new(Alias<std::type_identity<T>>, Alias<size_t>, std::align_val_t); // #20
  // clangext-warning@#20 {{type aware allocators are a Clang extension}}
  void *operator new(Alias<std::type_identity<int>>, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  template <typename T> void operator delete(Alias<std::type_identity<T>>, void *, size_t, std::align_val_t); // #21
  // clangext-warning@#21{{type aware allocators are a Clang extension}}
  void operator delete(Alias<std::type_identity<int>>, void *, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
};
struct S10 {
  template <typename T> void *operator new(TypeIdentityAlias<T>, size_t, std::align_val_t); // #22
  // clangext-warning@#22 {{type aware allocators are a Clang extension}}
  void *operator new(TypeIdentityAlias<int>, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  template <typename T> void operator delete(TypeIdentityAlias<T>, void *, size_t, std::align_val_t); // #23
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  void operator delete(TypeIdentityAlias<int>, void *, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
};

void test() {
  S9 *s9 = new S9;
  delete s9;
  S10 *s10 = new S10;
  delete s10;
}

struct S11 {
  template <typename T> void *operator new(TypedefAlias, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  void *operator new(TypedefAlias, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  template <typename T> void operator delete(TypedefAlias, void *, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  void operator delete(TypedefAlias, void *, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
};
struct S12 {
  template <typename T> void *operator new(UsingAlias, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  void *operator new(UsingAlias, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  template <typename T> void operator delete(UsingAlias, void *, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  void operator delete(UsingAlias, void *, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
};

struct S13 {
  void *operator new(std::type_identity<S13>, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
  void operator delete(std::type_identity<S13>, void*, size_t, std::align_val_t);
  // clangext-warning@-1 {{type aware allocators are a Clang extension}}
};
