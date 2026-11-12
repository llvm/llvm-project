// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

template<template<template<typename> class, typename> class T, template<typename> class V> struct PartialApply {
  template<typename W> using R = T<V, W>;
};

template<typename T> using Id = T;
template<template<typename> class, typename X> using Zero = X;
template<template<template<typename> class, typename> class N, template<typename> class F, typename X> using Succ = F<N<F,X>>;

template<template<typename> class F, typename X> using One = Succ<Zero, F, X>;
template<template<typename> class F, typename X> using Two = Succ<One, F, X>;

template<template<template<typename> class, typename> class A,
         template<template<typename> class, typename> class B,
         template<typename> class F,
         typename X> using Add = A<F, B<F, X>>;

template<template<template<typename> class, typename> class A,
         template<template<typename> class, typename> class B,
         template<typename> class F,
         typename X> using Mul = A<PartialApply<B,F>::template R, X>;

template<template<typename> class F, typename X> using Four = Add<Two, Two, F, X>;
template<template<typename> class F, typename X> using Sixteen = Mul<Four, Four, F, X>;
template<template<typename> class F, typename X> using TwoHundredAndFiftySix = Mul<Sixteen, Sixteen, F, X>;

template<typename T, T N> struct Const { static const T value = N; };
template<typename A> struct IncrementHelper;
template<typename T, T N> struct IncrementHelper<Const<T, N>> { using Result = Const<T, N+1>; };
template<typename A> using Increment = typename IncrementHelper<A>::Result;

using Arr = int[TwoHundredAndFiftySix<Increment, Const<int, 0>>::value];
using Arr = int[256];
