@import cxx_templates_common;

template<typename T> T f();
template<typename T> T f(T t) { return t; }
namespace N {
  template<typename T> T f();
  template<typename T> T f(T t) { return t; }
}

template<typename> int template_param_kinds_1();
template<template<typename, int, int...> class> int template_param_kinds_2();
template<template<typename T, typename U, U> class> int template_param_kinds_3();

template<typename T> struct SomeTemplate<T&> {};
template<typename T> struct SomeTemplate<T&>;
typedef SomeTemplate<int&> SomeTemplateIntRef;

extern DefinedInCommon &defined_in_common;

template<int> struct MergeTemplates;
MergeTemplates<0> *merge_templates_b;

template<typename T> template<typename U>
constexpr int Outer<T>::Inner<U>::g() { return 2; }
static_assert(Outer<int>::Inner<int>::g() == 2, "");

namespace TestInjectedClassName {
  template<typename T> struct X { X(); };
  typedef X<char[2]> B;
}

@import cxx_templates_b_impl;

template<typename T, typename> struct Identity { typedef T type; };
template<typename T> void UseDefinedInBImpl() {
  typename Identity<DefinedInBImpl, T>::type dependent;
  FoundByADL(dependent);
  typename Identity<DefinedInBImpl, T>::type::Inner inner;
  dependent.f();
}

extern DefinedInBImpl &defined_in_b_impl;

template<typename T>
struct RedeclareTemplateAsFriend {
  template<typename U>
  friend struct RedeclaredAsFriend;
};

void use_some_template_b() {
  SomeTemplate<char[1]> a;
  SomeTemplate<char[2]> b, c;
  b = c;

  WithImplicitSpecialMembers<int> wism1, wism2(wism1);
}

auto enum_b_from_b = CommonTemplate<int>::b;
const auto enum_c_from_b = CommonTemplate<int>::c;

template<int> struct UseInt;
template<typename T> void UseRedeclaredEnum(UseInt<T() + CommonTemplate<char>::a>);
constexpr void (*UseRedeclaredEnumB)(UseInt<1>) = UseRedeclaredEnum<int>;

typedef WithPartialSpecialization<void(int)>::type WithPartialSpecializationInstantiate3;

template<typename> struct MergeSpecializations;
template<typename T> struct MergeSpecializations<T&> {
  typedef int partially_specialized_in_b;
};
template<> struct MergeSpecializations<double> {
  typedef int explicitly_specialized_in_b;
};

template<typename U> using AliasTemplate = U;

void InstantiateWithAliasTemplate(WithAliasTemplate<int>::X<char>);
inline int InstantiateWithAnonymousDeclsB(WithAnonymousDecls<int> x) {
  return (x.k ? x.a : x.b) + (x.k ? x.s.c : x.s.d) + x.e;
}
inline int InstantiateWithAnonymousDeclsB2(WithAnonymousDecls<char> x) {
  return (x.k ? x.a : x.b) + (x.k ? x.s.c : x.s.d) + x.e;
}

@import cxx_templates_a;
template<typename T> void UseDefinedInBImplIndirectly(T &v) {
  PerformDelayedLookup(v);
}

void TriggerInstantiation() {
  UseDefinedInBImpl<void>();
  Std::f<int>();
  PartiallyInstantiatePartialSpec<int*>::foo();
  WithPartialSpecialization<void(int)>::type x;
}
