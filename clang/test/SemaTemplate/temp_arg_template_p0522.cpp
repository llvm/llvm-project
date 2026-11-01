// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// expected-note@temp_arg_template_p0522.cpp:* 1+{{template is declared here}}
// expected-note@temp_arg_template_p0522.cpp:* 1+{{template parameter is declared here}}
// expected-note@temp_arg_template_p0522.cpp:* 1+{{previous template template parameter is here}}

template<template<int> typename> struct Ti; // #Ti
template<template<int...> typename> struct TPi; // #TPi
template<template<int, int...> typename> struct TiPi;
template<template<int..., int...> typename> struct TPiPi;
// expected-error@-1 {{template parameter pack must be the last template parameter}}

template<typename T, template<T> typename> struct tT0; // #tT0
template<template<typename T, T> typename> struct Tt0; // #Tt0

template<template<typename> typename> struct Tt;
template<template<typename, typename...> typename> struct TtPt;

template<int> struct i;
template<int, int = 0> struct iDi;
template<int, int> struct ii;
template<int...> struct Pi;
template<int, int, int...> struct iiPi;

template<int, typename = int> struct iDt; // #iDt
template<int, typename> struct it; // #it

template<typename T, T v> struct t0;

template<typename...> struct Pt;

namespace IntParam {
  using ok = Pt<Ti<i>,
        Ti<iDi>,
        Ti<Pi>,
        Ti<iDt>>;
  using err1 = Ti<ii>; // expected-error {{too few template arguments for class template 'ii'}}
                       // expected-note@-1 {{different template parameters}}
  using err2 = Ti<iiPi>; // expected-error {{too few template arguments for class template 'iiPi'}}
                         // expected-note@-1 {{different template parameters}}
  using err3 = Ti<t0>; // expected-error@#Ti {{template argument for template type parameter must be a type}}
                       // expected-note@-1 {{different template parameters}}
  using err4 = Ti<it>; // expected-error {{too few template arguments for class template 'it'}}
                       // expected-note@-1 {{different template parameters}}
}

// These are accepted by the backwards-compatibility "parameter pack in
// parameter matches any number of parameters in arguments" rule.
namespace IntPackParam {
  using ok = TPi<Pi>;
  using ok_compat = Pt<TPi<i>, TPi<iDi>, TPi<ii>, TPi<iiPi>>;
  using err1 = TPi<t0>; // expected-error@#TPi {{template argument for template type parameter must be a type}}
                        // expected-note@-1 {{different template parameters}}
  using err2 = TPi<iDt>; // expected-error@#TPi {{template argument for template type parameter must be a type}}
                         // expected-note@-1 {{different template parameters}}
  using err3 = TPi<it>; // expected-error@#TPi {{template argument for template type parameter must be a type}}
                        // expected-note@-1 {{different template parameters}}
}

namespace IntAndPackParam {
  using ok = TiPi<Pi>;
  using ok_compat = Pt<TiPi<ii>, TiPi<iDi>, TiPi<iiPi>>;
  using err = TiPi<iDi>;
}

namespace DependentType {
  using ok = Pt<tT0<int, i>, tT0<int, iDi>>;
  using err1 = tT0<int, ii>; // expected-error {{too few template arguments for class template 'ii'}}
                             // expected-note@-1 {{different template parameters}}
  using err2 = tT0<short, i>;
  using err2a = tT0<long long, i>; // expected-error@#tT0 {{cannot be narrowed from type 'long long' to 'int'}}
                                   // expected-note@-1 {{different template parameters}}
  using err2b = tT0<void*, i>; // expected-error@#tT0 {{value of type 'void *' is not implicitly convertible to 'int'}}
                               // expected-note@-1 {{different template parameters}}
  using err3 = tT0<short, t0>; // expected-error@#tT0 {{template argument for template type parameter must be a type}}
                               // expected-note@-1 {{different template parameters}}

  using ok2 = Tt0<t0>;
  using err4 = Tt0<it>; // expected-error@#Tt0 {{template argument for non-type template parameter must be an expression}}
                        // expected-note@-1 {{different template parameters}}
}

namespace Auto {
  template<template<int> typename T> struct TInt {}; // #TInt
  template<template<int*> typename T> struct TIntPtr {}; // #TIntPtr
  template<template<auto> typename T> struct TAuto {}; // #TAuto
  template<template<auto*> typename T> struct TAutoPtr {};
  template<template<decltype(auto)> typename T> struct TDecltypeAuto {}; // #TDecltypeAuto
  template<auto> struct Auto;
  template<auto*> struct AutoPtr;
  template<decltype(auto)> struct DecltypeAuto;
  template<int> struct Int;
  template<int*> struct IntPtr;

  TInt<Auto> ia;
  TInt<AutoPtr> iap; // expected-error@#TInt {{non-type template parameter '' with type 'auto *' has incompatible initializer of type 'int'}}
                     // expected-note@-1 {{different template parameters}}
  TInt<DecltypeAuto> ida;
  TInt<Int> ii;
  TInt<IntPtr> iip; // expected-error@#TInt {{conversion from 'int' to 'int *' is not allowed in a converted constant expression}}
                    // expected-note@-1 {{different template parameters}}

  TIntPtr<Auto> ipa;
  TIntPtr<AutoPtr> ipap;
  TIntPtr<DecltypeAuto> ipda;
  TIntPtr<Int> ipi; // expected-error@#TIntPtr {{value of type 'int *' is not implicitly convertible to 'int'}}
                    // expected-note@-1 {{different template parameters}}
  TIntPtr<IntPtr> ipip;

  TAuto<Auto> aa;
  TAuto<AutoPtr> aap; // expected-error@#TAuto {{non-type template parameter '' with type 'auto *' has incompatible initializer of type 'auto'}}
                      // expected-note@-1 {{different template parameters}}
  TAuto<Int> ai; // FIXME: ill-formed (?)
  TAuto<IntPtr> aip; // FIXME: ill-formed (?)

  TAutoPtr<Auto> apa;
  TAutoPtr<AutoPtr> apap;
  TAutoPtr<Int> api; // FIXME: ill-formed (?)
  TAutoPtr<IntPtr> apip; // FIXME: ill-formed (?)

  TDecltypeAuto<DecltypeAuto> dada;
  TDecltypeAuto<Int> dai; // FIXME: ill-formed (?)
  TDecltypeAuto<IntPtr> daip; // FIXME: ill-formed (?)

  // FIXME: It's completely unclear what should happen here, but these results
  // seem at least plausible:
  TAuto<DecltypeAuto> ada;
  TAutoPtr<DecltypeAuto> apda;
  // Perhaps this case should be invalid, as there are valid 'decltype(auto)'
  // parameters (such as 'user-defined-type &') that are not valid 'auto'
  // parameters.
  TDecltypeAuto<Auto> daa;
  TDecltypeAuto<AutoPtr> daap; // expected-error@#TDecltypeAuto {{non-type template parameter '' with type 'auto *' has incompatible initializer of type 'decltype(auto)'}}
                               // expected-note@-1 {{different template parameters}}

  int n;
  template<auto A, decltype(A) B = &n> struct SubstFailure;
  TInt<SubstFailure> isf; // FIXME: this should be ill-formed
  TIntPtr<SubstFailure> ipsf;
}

namespace GH62529 {
  // Note: the constraint here is just for bypassing a fast-path.
  template<class T1> requires(true) using A = int;
  template<template<class ...T2s> class TT1, class T3> struct B {};
  template<class T4> B<A, T4> f();
  auto t = f<int>();
} // namespace GH62529

namespace GH101394 {
  struct X {}; // #X
  struct Y {
    constexpr Y(const X &) {}
  };

  namespace t1 {
    template<template<X> class> struct A {};
    template<Y> struct B;
    template struct A<B>;
  } // namespace t1
  namespace t2 {
    template<template<Y> class> struct A {}; // #A
    template<X> struct B; // #B
    template struct A<B>;
    // expected-error@#A {{no viable conversion from 'const Y' to 'X'}}
    // expected-note@-2  {{different template parameters}}
    // expected-note@#X 2{{not viable}}
    // expected-note@#B  {{passing argument to parameter here}}
  } // namespace t2
} // namespace GH101394

namespace GH181166 {
  template <template <class...> class> struct A;
  template <template <class...> class... TT1> A<TT1...> f();
  template <class ...Ts> struct B {};
  using T = decltype(f<B>());
} // namespace GH181166

// Matching of template template arguments whose constant parameters have
// reference type. The argument value is an expression, so its type drops the
// top-level reference (`auto &` is seen as `auto`).
//
// Names spell out the types of constant template parameters: `TakesFoo` has a
// template template parameter P whose parameter has type `Foo`, and `Foo_Bar`
// matches P against a template template argument A whose parameter has type
// `Bar`. A is a template template parameter, a class template `BarClass`, or a
// member template `MemberBar`.
namespace nttp_ref {
  template <template <auto &> class> struct TakesAutoRef; // #TakesAutoRef
  template <template <auto &> class TT> using AutoRef_AutoRef = TakesAutoRef<TT>;
  template <template <auto> class TT> using AutoRef_Auto = TakesAutoRef<TT>;
  template <template <auto &&> class TT> using AutoRef_AutoRRef = TakesAutoRef<TT>;

  template <template <auto> class> struct TakesAuto; // #TakesAuto
  template <template <auto &> class TT> using Auto_AutoRef = TakesAuto<TT>;
  template <template <auto &&> class TT> using Auto_AutoRRef = TakesAuto<TT>;
  template <template <const auto &> class TT> using Auto_ConstAutoRef = TakesAuto<TT>;

  template <template <const auto &> class> struct TakesConstAutoRef; // #TakesConstAutoRef
  template <template <const auto &> class TT> using ConstAutoRef_ConstAutoRef = TakesConstAutoRef<TT>;
  template <template <auto &> class TT> using ConstAutoRef_AutoRef = TakesConstAutoRef<TT>;
  template <template <auto> class TT> using ConstAutoRef_Auto = TakesConstAutoRef<TT>;

  template <template <auto &...> class> struct TakesAutoRefPack;
  template <template <auto &...> class TT> using AutoRefPack_AutoRefPack = TakesAutoRefPack<TT>;

  template <template <decltype(auto)> class> struct TakesDecltypeAuto; // #TakesDecltypeAuto
  template <template <decltype(auto)> class TT> using DecltypeAuto_DecltypeAuto = TakesDecltypeAuto<TT>;

  // A class template argument has its parameters at the same depth as those
  // of the template template parameter.
  template <auto &> struct AutoRefClass;
  template <auto &&> struct AutoRRefClass;
  template <auto> struct AutoClass;
  using AutoRef_AutoRefClass = TakesAutoRef<AutoRefClass>;
  using AutoRef_AutoRRefClass = TakesAutoRef<AutoRRefClass>;
  using Auto_AutoRefClass = TakesAuto<AutoRefClass>;
  using AutoRef_AutoClass = TakesAutoRef<AutoClass>;

  template <class U, template <U &> class> struct TakesDependentRef;
  template <class U, template <U &> class TT> using DependentRef_DependentRef = TakesDependentRef<U, TT>;

  template <class U> struct Outer {
    template <template <U &> class> struct TakesOuterRef;
    template <template <U &> class TT> using OuterRef_OuterRef = TakesOuterRef<TT>;
  };

  template <template <class T, T &> class> struct TakesTypeAndRef;
  template <template <class T, T &> class TT> using TypeAndRef_TypeAndRef = TakesTypeAndRef<TT>;
  template <template <class T, T> class TT> using TypeAndRef_TypeAndValue = TakesTypeAndRef<TT>;
  template <class T, T &> struct TypeAndRefClass;
  using TypeAndRef_TypeAndRefClass = TakesTypeAndRef<TypeAndRefClass>;

  // The argument for P's parameter must be a valid argument for A's parameter.
  template <template <int> class> struct TakesInt; // #TakesInt
  // expected-error@-1 {{value of type 'int' is not implicitly convertible to 'int &'}}
  template <template <int &> class TT> using Int_IntRef = TakesInt<TT>;
  // expected-note@-1 {{different template parameters}}
  template <template <const int &> class TT> using Int_ConstIntRef = TakesInt<TT>;
  // expected-error@#TakesInt {{conversion from 'int' to 'const int &' in converted constant expression would bind reference to a temporary}}
  // expected-note@-2 {{different template parameters}}
  template <template <auto &> class TT> using Int_AutoRef = TakesInt<TT>;
  // expected-error@#TakesInt {{value of type 'int' is not implicitly convertible to 'int &'}}
  // expected-note@-2 {{different template parameters}}
  using Int_AutoRefClass = TakesInt<AutoRefClass>;
  // expected-error@#TakesInt {{value of type 'int' is not implicitly convertible to 'int &'}}
  // expected-note@-2 {{different template parameters}}
  template <template <auto &&> class TT> using Int_AutoRRef = TakesInt<TT>;
  // expected-error@#TakesInt {{non-type template parameter has rvalue reference type 'int &&'}}
  // expected-note@-2 {{different template parameters}}
  template <template <const auto &> class TT> using Int_ConstAutoRef = TakesInt<TT>;
  // expected-error@#TakesInt {{conversion from 'int' to 'const int &' in converted constant expression would bind reference to a temporary}}
  // expected-note@-2 {{different template parameters}}

  template <template <int...> class> struct TakesIntPack; // #TakesIntPack
  template <template <auto &...> class TT> using IntPack_AutoRefPack = TakesIntPack<TT>;
  // expected-error@#TakesIntPack {{value of type 'int' is not implicitly convertible to 'int &'}}
  // expected-note@-2 {{different template parameters}}

  template <template <const int &> class> struct TakesConstIntRef; // #TakesConstIntRef
  template <template <int &> class TT> using ConstIntRef_IntRef = TakesConstIntRef<TT>;
  // expected-error@#TakesConstIntRef {{value of type 'const int' is not implicitly convertible to 'int &'}}
  // expected-note@-2 {{different template parameters}}

  template <template <int *> class> struct TakesIntPtr; // #TakesIntPtr
  template <template <int> class TT> using IntPtr_Int = TakesIntPtr<TT>;
  // expected-error@#TakesIntPtr {{value of type 'int *' is not implicitly convertible to 'int'}}
  // expected-note@-2 {{different template parameters}}
  template <template <auto &> class TT> using IntPtr_AutoRef = TakesIntPtr<TT>;
  // expected-error@#TakesIntPtr {{value of type 'int *' is not implicitly convertible to 'int *&'}}
  // expected-note@-2 {{different template parameters}}

  // Only a pointer can initialize a parameter of type `auto *`.
  template <template <auto *> class TT> using Int_AutoPtr = TakesInt<TT>;
  // expected-error@#TakesInt {{with type 'auto *' has incompatible initializer of type 'int'}}
  // expected-note@-2 {{different template parameters}}
  template <template <auto *> class TT> using AutoRef_AutoPtr = TakesAutoRef<TT>;
  // expected-error@#TakesAutoRef {{with type 'auto *' has incompatible initializer of type 'auto'}}
  // expected-note@-2 {{different template parameters}}
  template <template <auto *> class TT> using Auto_AutoPtr = TakesAuto<TT>;
  // expected-error@#TakesAuto {{with type 'auto *' has incompatible initializer of type 'auto'}}
  // expected-note@-2 {{different template parameters}}
  template <template <auto *> class TT> using ConstAutoRef_AutoPtr = TakesConstAutoRef<TT>;
  // expected-error@#TakesConstAutoRef {{with type 'auto *' has incompatible initializer of type 'const auto'}}
  // expected-note@-2 {{different template parameters}}
  template <template <auto *> class TT> using DecltypeAuto_AutoPtr = TakesDecltypeAuto<TT>;
  // expected-error@#TakesDecltypeAuto {{with type 'auto *' has incompatible initializer of type 'decltype(auto)'}}
  // expected-note@-2 {{different template parameters}}
  template <class> struct DependentAutoPtr {
    template <auto *> struct MemberAutoPtr;
    using Auto_MemberAutoPtr = TakesAuto<MemberAutoPtr>;
    // expected-error@#TakesAuto {{with type 'auto *' has incompatible initializer of type 'auto'}}
    // expected-note@-2 {{different template parameters}}
  };
} // namespace nttp_ref
