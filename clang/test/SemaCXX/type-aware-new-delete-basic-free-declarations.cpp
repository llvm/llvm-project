// RUN: %clang_cc1 -fsyntax-only -verify %s          -std=c++23 -fexperimental-cxx-type-aware-allocators
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TAA -std=c++23 -fno-experimental-cxx-type-aware-allocators

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;

struct TestType {};
template <typename T> struct TemplateTestType {};

// Valid free declarations
void *operator new(std::type_identity<int>, size_t); // #1
void *operator new(std::type_identity<int>, size_t, std::align_val_t); // #2
void *operator new(std::type_identity<int>, size_t, TestType&); // #3
template <typename T> void *operator new(std::type_identity<T>, size_t); // #4
template <typename T> void *operator new(std::type_identity<T>, size_t, TestType&); // #5
template <typename T> void *operator new(std::type_identity<TemplateTestType<T>>, size_t, TestType&); // #6
template <typename T, typename U> void *operator new(std::type_identity<T>, size_t, TemplateTestType<U>&); // #7
template <template <typename> class T> void *operator new(std::type_identity<T<int>>, size_t); // #8
#if defined(NO_TAA)
//expected-error@#1 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#2 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#3 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#4 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#5 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#6 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#7 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#8 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

void operator delete(std::type_identity<int>, void *); // #9
void operator delete(std::type_identity<int>, void *, std::align_val_t); // #10
void operator delete(std::type_identity<int>, void *, size_t); // #11
void operator delete(std::type_identity<int>, void *, size_t, std::align_val_t); // #12
#if defined(NO_TAA)
//expected-error@#9 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#10 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#11 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#12 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

template <typename T> void operator delete(std::type_identity<T>, void *); // #13
template <typename T> void operator delete(std::type_identity<T>, void *, std::align_val_t); // #14
template <typename T> void operator delete(std::type_identity<T>, void *, size_t); // #15
template <typename T> void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t); // #16
#if defined(NO_TAA)
//expected-error@#13 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#14 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#15 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#16 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

template <typename T> void operator delete(std::type_identity<TemplateTestType<T>>, void *); // #17
template <template <typename> class T> void operator delete(std::type_identity<T<int>>, void *); // #18
#if defined(NO_TAA)
//expected-error@#17 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
//expected-error@#18 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

typedef std::type_identity<float> TypeIdentityAlias1;
void *operator new(TypeIdentityAlias1, size_t); // #19
#if defined(NO_TAA)
//expected-error@#19 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

using TypeIdentityAlias2 = std::type_identity<double>;
void *operator new(TypeIdentityAlias2, size_t); // #20
#if defined(NO_TAA)
//expected-error@#20 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif

template <typename T> using TypeIdentityAlias3 = std::type_identity<T>;
template <typename T> void *operator new(TypeIdentityAlias3<T>, size_t); // #21
#if defined(NO_TAA)
//expected-error@#21 {{type aware allocation operators are disabled, enable with '-fexperimental-cxx-type-aware-allocators'}}
#endif


template <typename T> void *operator new(T, size_t);
// expected-error@-1 {{'operator new' cannot take a dependent type as first parameter}}

template <typename T> void operator delete(T, void*);
// expected-error@-1 {{'operator delete' cannot take a dependent type as first parameter}}

template <typename T> struct S {
  typedef std::type_identity<T> type_identity;
  typedef size_t size_ty;
  typedef void *ptr_ty;
};

template <typename T> void *operator new(typename S<T>::type_identity, size_t);
// expected-error@-1 {{'operator new' cannot take a dependent type as first parameter}}

#if !defined(NO_TAA)
template <typename U> void *operator new(std::type_identity<int>, U);
// expected-error@-1 {{type aware 'operator new' cannot take a dependent type as second parameter}}
template <typename U> void operator delete(std::type_identity<int>, U);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as second parameter}}
template <typename T, typename U> void *operator new(std::type_identity<T>, U);
// expected-error@-1 {{type aware 'operator new' cannot take a dependent type as second parameter}}
template <typename T, typename U> void operator delete(std::type_identity<T>, U);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as second parameter}}
template <typename U> void *operator new(std::type_identity<int>, typename S<U>::size_ty);
// expected-error@-1 {{type aware 'operator new' cannot take a dependent type as second parameter}}
template <typename U> void operator delete(std::type_identity<int>, typename S<U>::ptr_ty);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as second parameter}}
template <typename T, typename U> void *operator new(std::type_identity<T>, typename S<U>::size_ty);
// expected-error@-1 {{type aware 'operator new' cannot take a dependent type as second parameter}}
template <typename T, typename U> void operator delete(std::type_identity<T>, typename S<U>::ptr_ty);
// expected-error@-1 {{type aware 'operator delete' cannot take a dependent type as second parameter}}
#endif
