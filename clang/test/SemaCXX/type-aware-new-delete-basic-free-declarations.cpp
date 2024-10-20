// RUN: %clang_cc1 -fsyntax-only -verify %s          -std=c++17 -fexperimental-cxx-type-aware-allocators
// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TAA -std=c++17 -fno-experimental-cxx-type-aware-allocators

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;

struct TestType {};
template <typename T> struct TemplateTestType {};

// Valid free declarations
void *operator new(std::type_identity<int>, size_t);
void *operator new(std::type_identity<int>, size_t, std::align_val_t);
void *operator new(std::type_identity<int>, size_t, TestType&);
template <typename T> void *operator new(std::type_identity<T>, size_t);
template <typename T> void *operator new(std::type_identity<T>, size_t, TestType&);
template <typename T> void *operator new(std::type_identity<TemplateTestType<T>>, size_t, TestType&);
template <typename T, typename U> void *operator new(std::type_identity<T>, size_t, TemplateTestType<U>&);
template <template <typename> class T> void *operator new(std::type_identity<T<int>>, size_t); 
#if defined(NO_TAA)
//expected-error@-9 {{type aware allocation operators are disabled}}
//expected-error@-9 {{type aware allocation operators are disabled}}
//expected-error@-9 {{type aware allocation operators are disabled}}
//expected-error@-9 {{type aware allocation operators are disabled}}
//expected-error@-9 {{type aware allocation operators are disabled}}
//expected-error@-9 {{type aware allocation operators are disabled}}
//expected-error@-9 {{type aware allocation operators are disabled}}
//expected-error@-9 {{type aware allocation operators are disabled}}
#endif

void operator delete(std::type_identity<int>, void *);
void operator delete(std::type_identity<int>, void *, std::align_val_t);
void operator delete(std::type_identity<int>, void *, size_t);
void operator delete(std::type_identity<int>, void *, size_t, std::align_val_t);
#if defined(NO_TAA)
//expected-error@-5 {{type aware allocation operators are disabled}}
//expected-error@-5 {{type aware allocation operators are disabled}}
//expected-error@-5 {{type aware allocation operators are disabled}}
//expected-error@-5 {{type aware allocation operators are disabled}}
#endif

template <typename T> void operator delete(std::type_identity<T>, void *);
template <typename T> void operator delete(std::type_identity<T>, void *, std::align_val_t);
template <typename T> void operator delete(std::type_identity<T>, void *, size_t);
template <typename T> void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t);
#if defined(NO_TAA)
//expected-error@-5 {{type aware allocation operators are disabled}}
//expected-error@-5 {{type aware allocation operators are disabled}}
//expected-error@-5 {{type aware allocation operators are disabled}}
//expected-error@-5 {{type aware allocation operators are disabled}}
#endif

template <typename T> void operator delete(std::type_identity<TemplateTestType<T>>, void *);
template <template <typename> class T> void operator delete(std::type_identity<T<int>>, void *);
#if defined(NO_TAA)
//expected-error@-3 {{type aware allocation operators are disabled}}
//expected-error@-3 {{type aware allocation operators are disabled}}
#endif

typedef std::type_identity<float> TypeIdentityAlias1;
void *operator new(TypeIdentityAlias1, size_t);
#if defined(NO_TAA)
//expected-error@-2 {{type aware allocation operators are disabled}}
#endif

using TypeIdentityAlias2 = std::type_identity<double>;
void *operator new(TypeIdentityAlias2, size_t);
#if defined(NO_TAA)
//expected-error@-2 {{type aware allocation operators are disabled}}
#endif

template <typename T> using TypeIdentityAlias3 = std::type_identity<T>;
template <typename T> void *operator new(TypeIdentityAlias3<T>, size_t);
#if defined(NO_TAA)
//expected-error@-2 {{type aware allocation operators are disabled}}
#endif


// Invalid free declarations - need to update error text
template <typename T> void *operator new(T, size_t);
// expected-error@-1 {{'operator new' cannot take a dependent type as first parameter; use size_t ('unsigned long') instead}}

template <typename T> void operator delete(T, void*);
// expected-error@-1 {{'operator delete' cannot take a dependent type as first parameter; use 'void *' instead}}

template <typename T> struct S {
  typedef std::type_identity<T> type_identity;
};

template <typename T> void *operator new(typename S<T>::type_identity, size_t);
// expected-error@-1 {{'operator new' cannot take a dependent type as first parameter; use size_t ('unsigned long') instead}}
