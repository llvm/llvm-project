// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// expected-note@temp_arg_template_p0522.cpp:* 1+{{}}

template<template<int> typename> struct Ti;
template<template<int...> typename> struct TPi;
template<template<int, int...> typename> struct TiPi;
template<template<int..., int...> typename> struct TPiPi; // FIXME: Why is this not ill-formed?

template<typename T, template<T> typename> struct tT0;
template<template<typename T, T> typename> struct Tt0;

template<template<typename> typename> struct Tt;
template<template<typename, typename...> typename> struct TtPt;

template<int> struct i;
template<int, int = 0> struct iDi;
template<int, int> struct ii;
template<int...> struct Pi;
template<int, int, int...> struct iiPi;

template<int, typename = int> struct iDt;
template<int, typename> struct it;

template<typename T, T v> struct t0;

template<typename...> struct Pt;

namespace IntParam {
  using ok = Pt<Ti<i>,
        Ti<iDi>,
        Ti<Pi>,
        Ti<iDt>>;
  using err1 = Ti<ii>; // expected-error {{different template parameters}}
  using err2 = Ti<iiPi>; // expected-error {{different template parameters}}
  using err3 = Ti<t0>; // expected-error {{different template parameters}}
  using err4 = Ti<it>; // expected-error {{different template parameters}}
}

// These are accepted by the backwards-compatibility "parameter pack in
// parameter matches any number of parameters in arguments" rule.
namespace IntPackParam {
  using ok = TPi<Pi>;
  using ok_compat = Pt<TPi<i>, TPi<iDi>, TPi<ii>, TPi<iiPi>>;
  using err1 = TPi<t0>; // expected-error {{different template parameters}}
  using err2 = TPi<iDt>; // expected-error {{different template parameters}}
  using err3 = TPi<it>; // expected-error {{different template parameters}}
}

namespace IntAndPackParam {
  using ok = TiPi<Pi>;
  using ok_compat = Pt<TiPi<ii>, TiPi<iDi>, TiPi<iiPi>>;
  using err = TiPi<iDi>;
}

namespace DependentType {
  using ok = Pt<tT0<int, i>, tT0<int, iDi>>;
  using err1 = tT0<int, ii>; // expected-error {{different template parameters}}
  using err2 = tT0<short, i>; // FIXME: should this be OK?
  using err2a = tT0<long long, i>; // FIXME: should this be OK (if long long is larger than int)?
  using err2b = tT0<void*, i>; // expected-error {{different template parameters}}
  using err3 = tT0<short, t0>; // expected-error {{different template parameters}}

  using ok2 = Tt0<t0>;
  using err4 = Tt0<it>; // expected-error {{different template parameters}}
}

namespace Auto {
  template<template<int> typename T> struct TInt {};
  template<template<int*> typename T> struct TIntPtr {};
  template<template<auto> typename T> struct TAuto {};
  template<template<auto*> typename T> struct TAutoPtr {};
  template<template<decltype(auto)> typename T> struct TDecltypeAuto {};
  template<auto> struct Auto;
  template<auto*> struct AutoPtr;
  template<decltype(auto)> struct DecltypeAuto;
  template<int> struct Int;
  template<int*> struct IntPtr;

  TInt<Auto> ia;
  TInt<AutoPtr> iap; // FIXME: ill-formed (?)
  TInt<DecltypeAuto> ida;
  TInt<Int> ii;
  TInt<IntPtr> iip; // expected-error {{different template parameters}}

  TIntPtr<Auto> ipa;
  TIntPtr<AutoPtr> ipap;
  TIntPtr<DecltypeAuto> ipda;
  TIntPtr<Int> ipi; // expected-error {{different template parameters}}
  TIntPtr<IntPtr> ipip;

  TAuto<Auto> aa;
  TAuto<AutoPtr> aap; // FIXME: ill-formed (?)
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
  TDecltypeAuto<AutoPtr> daap; // FIXME: should probably be ill-formed

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
