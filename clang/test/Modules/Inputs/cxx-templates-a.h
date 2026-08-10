@import cxx_templates_common;

template<typename T> T f() { return T(); }
template<typename T> T f(T);
namespace N {
  template<typename T> T f() { return T(); }
  template<typename T> T f(T);
}

template<int N> int template_param_kinds_1();
template<template<typename T, int, int> class> int template_param_kinds_2();
template<template<typename T, typename U, T> class> int template_param_kinds_3();

template<typename T> struct SomeTemplate<T*>;
template<typename T> struct SomeTemplate<T*> {};
typedef SomeTemplate<int*> SomeTemplateIntPtr;

template<typename T> void PerformDelayedLookup(T &t) {
  t.f();
  typename T::Inner inner;
  FoundByADL(t);
}

template<typename T> void PerformDelayedLookupInDefaultArgument(T &t, int a = (FoundByADL(T()), 0)) {}

template<typename T> struct RedeclaredAsFriend {};

void use_some_template_a() {
  SomeTemplate<char[2]> a;
  SomeTemplate<char[1]> b, c;
  b = c;

  (void)&WithImplicitSpecialMembers<int>::n;
}

template<int> struct MergeTemplates;
MergeTemplates<0> *merge_templates_a;

auto enum_a_from_a = CommonTemplate<int>::a;
const auto enum_c_from_a = CommonTemplate<int>::c;

template<int> struct UseInt;
template<typename T> void UseRedeclaredEnum(UseInt<T() + CommonTemplate<char>::a>);
constexpr void (*UseRedeclaredEnumA)(UseInt<1>) = UseRedeclaredEnum<int>;

template<typename> struct MergeSpecializations;
template<typename T> struct MergeSpecializations<T*> {
  typedef int partially_specialized_in_a;
};
template<> struct MergeSpecializations<char> {
  typedef int explicitly_specialized_in_a;
};

void InstantiateWithFriend(Std::WithFriend<int> wfi) {}

template<typename T> struct WithPartialSpecialization<T*> {
  typedef int type;
  T &f() { static T t; return t; }
};
typedef WithPartialSpecializationUse::type WithPartialSpecializationInstantiate;
typedef WithPartialSpecialization<void(int)>::type WithPartialSpecializationInstantiate2;

template<> struct WithExplicitSpecialization<int> {
  int n;
  template<typename T> T &inner_template() {
    return n;
  }
};

template<typename T> template<typename U>
constexpr int Outer<T>::Inner<U>::f() { return 1; }
static_assert(Outer<int>::Inner<int>::f() == 1, "");

template<typename T> struct MergeTemplateDefinitions {
  static constexpr int f();
  static constexpr int g();
};
template<typename T> constexpr int MergeTemplateDefinitions<T>::f() { return 1; }

template<typename T> using AliasTemplate = T;

template<typename T> struct PartiallyInstantiatePartialSpec {};
template<typename T> struct PartiallyInstantiatePartialSpec<T*> {
  static T *foo() { return reinterpret_cast<T*>(0); }
  static T *bar() { return reinterpret_cast<T*>(0); }
};
typedef PartiallyInstantiatePartialSpec<int*> PartiallyInstantiatePartialSpecHelper;

void InstantiateWithAliasTemplate(WithAliasTemplate<int>::X<char>);
inline int InstantiateWithAnonymousDeclsA(WithAnonymousDecls<int> x) { return (x.k ? x.a : x.b) + (x.k ? x.s.c : x.s.d) + x.e; }
inline int InstantiateWithAnonymousDeclsB2(WithAnonymousDecls<char> x);


template<typename T1 = int>
struct MergeAnonUnionMember {
  MergeAnonUnionMember() { (void)values.t1; }
  union { int t1; } values;
};
inline MergeAnonUnionMember<> maum_a() { return {}; }

template<typename T> struct DontWalkPreviousDeclAfterMerging { struct Inner { typedef T type; }; };

namespace TestInjectedClassName {
  template<typename T> struct X { X(); };
  typedef X<char[1]> A;
}
