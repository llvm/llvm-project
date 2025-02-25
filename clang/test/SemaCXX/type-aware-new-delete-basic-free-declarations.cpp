// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++23 -fexperimental-cxx-type-aware-allocators    -fsized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++23 -fexperimental-cxx-type-aware-allocators -fno-sized-deallocation    -faligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++23 -fexperimental-cxx-type-aware-allocators -fno-sized-deallocation -fno-aligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s          -std=c++23 -fexperimental-cxx-type-aware-allocators    -fsized-deallocation -fno-aligned-allocation
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s -DNO_TAA -std=c++23 -fno-experimental-cxx-type-aware-allocators

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;

struct TestType {};
template <typename T> struct TemplateTestType {};


// Valid free declarations
void *operator new(std::type_identity<int>, size_t, std::align_val_t); // #1
void *operator new(std::type_identity<int>, size_t, std::align_val_t, TestType&); // #2
template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t); // #3
template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t, TestType&); // #4
template <typename T> void *operator new(std::type_identity<TemplateTestType<T>>, size_t, std::align_val_t, TestType&); // #5
template <typename T, typename U> void *operator new(std::type_identity<T>, size_t, std::align_val_t, TemplateTestType<U>&); // #6
template <template <typename> class T> void *operator new(std::type_identity<T<int>>, size_t, std::align_val_t); // #7
#if defined(NO_TAA)
//expected-error@#1 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#2 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#3 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#4 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#5 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#6 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#7 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

void operator delete(std::type_identity<int>, void *, size_t, std::align_val_t); // #8
template <typename T> void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t); // #9
template <typename T> void operator delete(std::type_identity<TemplateTestType<T>>, void *, size_t, std::align_val_t); // #10
template <template <typename> class T> void operator delete(std::type_identity<T<int>>, void *, size_t, std::align_val_t); // #11

#if defined(NO_TAA)
//expected-error@#8 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#9 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#10 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#11 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

typedef std::type_identity<float> TypeIdentityAlias1;
void *operator new(TypeIdentityAlias1, size_t, std::align_val_t); // #12
#if defined(NO_TAA)
//expected-error@#12 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

using TypeIdentityAlias2 = std::type_identity<double>;
void *operator new(TypeIdentityAlias2, size_t, std::align_val_t); // #13
#if defined(NO_TAA)
//expected-error@#13 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

template <typename T> using TypeIdentityAlias3 = std::type_identity<T>;
template <typename T> void *operator new(TypeIdentityAlias3<T>, size_t, std::align_val_t); // #14
#if defined(NO_TAA)
//expected-error@#14 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif


template <typename T> void *operator new(T, size_t, std::align_val_t);
// expected-error@-1 {{'operator new' cannot take a dependent type as its first parameter}}

template <typename T> void operator delete(T, void*, size_t, std::align_val_t);
// expected-error@-1 {{'operator delete' cannot take a dependent type as its first parameter}}

template <typename T> struct S {
  typedef std::type_identity<T> type_identity;
  typedef size_t size_ty;
  typedef std::align_val_t align_val_ty;
  typedef void *ptr_ty;
};

template <typename T> void *operator new(typename S<T>::type_identity, size_t, std::align_val_t);
// expected-error@-1 {{'operator new' cannot take a dependent type as its first parameter}}

#if !defined(NO_TAA)
// Invalid type aware declarations
void *operator new(std::type_identity<int>, size_t); 
// expected-error@-1 {{type aware 'operator new' must have at least three parameters}}
void *operator new(std::type_identity<int>, size_t, TestType&);
// expected-error@-1 {{type aware 'operator new' takes type std::align_val_t ('std::align_val_t') as third parameter}}
void operator delete(std::type_identity<int>, void *);
// expected-error@-1 {{type aware 'operator delete' must have at least four parameters}}
void operator delete(std::type_identity<int>, void *, size_t);
// expected-error@-1 {{type aware 'operator delete' must have at least four parameters}}
void operator delete(std::type_identity<int>, void *, std::align_val_t);
// expected-error@-1 {{type aware 'operator delete' must have at least four parameters}}
template <typename T> void operator delete(std::type_identity<T>, void *);
// expected-error@-1 {{type aware 'operator delete' must have at least four parameters}}
template <typename T> void operator delete(std::type_identity<T>, void *, std::align_val_t);
// expected-error@-1 {{type aware 'operator delete' must have at least four parameters}}
template <typename T> void operator delete(std::type_identity<T>, void *, size_t);
// expected-error@-1 {{type aware 'operator delete' must have at least four parameters}}
template <typename T, typename U> void *operator new(std::type_identity<T>, U);
// expected-error@-1 {{type aware 'operator new' must have at least three parameters}}
template <typename T, typename U> void operator delete(std::type_identity<T>, U, size_t, std::align_val_t);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as its second parameter; use 'void *' instead}}
template <typename T, typename U> void operator delete(std::type_identity<T>, void *, U, std::align_val_t);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as its third parameter; use 'unsigned long' instead}}
template <typename T, typename U> void operator delete(std::type_identity<T>, void *, size_t, U);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as its fourth parameter; use 'std::align_val_t' instead}}
template <typename U> void *operator new(std::type_identity<int>, typename S<U>::size_ty, std::align_val_t);
// expected-error@-1 {{type aware 'operator new' cannot take a dependent type as its second parameter; use size_t ('unsigned long') instead}}
template <typename U> void operator delete(std::type_identity<int>, typename S<U>::ptr_ty, size_t, std::align_val_t);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as its second parameter; use 'void *' instead}}
template <typename T, typename U> void *operator new(std::type_identity<T>, typename S<U>::size_ty, std::align_val_t);
// expected-error@-1 {{type aware 'operator new' cannot take a dependent type as its second parameter; use size_t ('unsigned long') instead}}
template <typename T, typename U> void operator delete(std::type_identity<T>, typename S<U>::ptr_ty, size_t, std::align_val_t);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as its second parameter; use 'void *' instead}}
template <typename T, typename U> void operator delete(std::type_identity<T>, void *, size_t, typename S<U>::align_val_ty);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as its fourth parameter; use 'std::align_val_t' instead}}

template <typename T> using Alias = T;
template <typename T> using TypeIdentityAlias = std::type_identity<T>;
typedef std::type_identity<double> TypedefAlias;
using UsingAlias = std::type_identity<float>;
void *operator new(Alias<size_t>, std::align_val_t);
template <typename T> void *operator new(Alias<std::type_identity<T>>, Alias<size_t>, std::align_val_t);
void *operator new(Alias<std::type_identity<int>>, size_t, std::align_val_t);
template <typename T> void operator delete(Alias<std::type_identity<T>>, void *, size_t, std::align_val_t);
void operator delete(Alias<std::type_identity<int>>, void *, size_t, std::align_val_t);

template <typename T> void *operator new(TypeIdentityAlias<T>, size_t, std::align_val_t);
void *operator new(TypeIdentityAlias<int>, size_t, std::align_val_t);
template <typename T> void operator delete(TypeIdentityAlias<T>, void *, size_t, std::align_val_t);
void operator delete(TypeIdentityAlias<int>, void *, size_t, std::align_val_t);

template <typename T> void *operator new(TypedefAlias, size_t, std::align_val_t);
void *operator new(TypedefAlias, size_t, std::align_val_t);
template <typename T> void operator delete(TypedefAlias, void *, size_t, std::align_val_t);
void operator delete(TypedefAlias, void *, size_t, std::align_val_t);

template <typename T> void *operator new(UsingAlias, size_t, std::align_val_t);
void *operator new(UsingAlias, size_t, std::align_val_t);
template <typename T> void operator delete(UsingAlias, void *, size_t, std::align_val_t);
void operator delete(UsingAlias, void *, size_t, std::align_val_t);

class ForwardDecl;
void *operator new(std::type_identity<ForwardDecl>, size_t, std::align_val_t);
void operator delete(std::type_identity<ForwardDecl>, void*, size_t, std::align_val_t);

#endif
