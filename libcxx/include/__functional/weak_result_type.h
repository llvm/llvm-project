// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FUNCTIONAL_WEAK_RESULT_TYPE_H
#define _LIBCPP___FUNCTIONAL_WEAK_RESULT_TYPE_H

#include <__config>
#include <__functional/binary_function.h>
#include <__functional/unary_function.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __has_result_type
{
private:
    template <class _Up> static false_type __test(...);
    template <class _Up> static true_type __test(typename _Up::result_type* = 0);
public:
    static const bool value = decltype(__test<_Tp>(0))::value;
};

// __weak_result_type

template <class _Tp>
struct __derives_from_unary_function
{
private:
    struct __two {char __lx; char __lxx;};
    static __two __test(...);
    template <class _Ap, class _Rp>
        static unary_function<_Ap, _Rp>
        __test(const volatile unary_function<_Ap, _Rp>*);
public:
    static const bool value = !is_same<decltype(__test((_Tp*)0)), __two>::value;
    typedef decltype(__test((_Tp*)0)) type;
};

template <class _Tp>
struct __derives_from_binary_function
{
private:
    struct __two {char __lx; char __lxx;};
    static __two __test(...);
    template <class _A1, class _A2, class _Rp>
        static binary_function<_A1, _A2, _Rp>
        __test(const volatile binary_function<_A1, _A2, _Rp>*);
public:
    static const bool value = !is_same<decltype(__test((_Tp*)0)), __two>::value;
    typedef decltype(__test((_Tp*)0)) type;
};

template <class _Tp, bool = __derives_from_unary_function<_Tp>::value>
struct __maybe_derive_from_unary_function  // bool is true
    : public __derives_from_unary_function<_Tp>::type
{
};

template <class _Tp>
struct __maybe_derive_from_unary_function<_Tp, false>
{
};

template <class _Tp, bool = __derives_from_binary_function<_Tp>::value>
struct __maybe_derive_from_binary_function  // bool is true
    : public __derives_from_binary_function<_Tp>::type
{
};

template <class _Tp>
struct __maybe_derive_from_binary_function<_Tp, false>
{
};

template <class _Tp, bool = __has_result_type<_Tp>::value>
struct __weak_result_type_imp // bool is true
    : public __maybe_derive_from_unary_function<_Tp>,
      public __maybe_derive_from_binary_function<_Tp>
{
    typedef _LIBCPP_NODEBUG typename _Tp::result_type result_type;
};

template <class _Tp>
struct __weak_result_type_imp<_Tp, false>
    : public __maybe_derive_from_unary_function<_Tp>,
      public __maybe_derive_from_binary_function<_Tp>
{
};

template <class _Tp>
struct __weak_result_type
    : public __weak_result_type_imp<_Tp>
{
};

// 0 argument case

template <class _Rp>
struct __weak_result_type<_Rp ()>
{
    typedef _LIBCPP_NODEBUG _Rp result_type;
};

template <class _Rp>
struct __weak_result_type<_Rp (&)()>
{
    typedef _LIBCPP_NODEBUG _Rp result_type;
};

template <class _Rp>
struct __weak_result_type<_Rp (*)()>
{
    typedef _LIBCPP_NODEBUG _Rp result_type;
};

// 1 argument case

template <class _Rp, class _A1>
struct __weak_result_type<_Rp (_A1)>
    : public unary_function<_A1, _Rp>
{
};

template <class _Rp, class _A1>
struct __weak_result_type<_Rp (&)(_A1)>
    : public unary_function<_A1, _Rp>
{
};

template <class _Rp, class _A1>
struct __weak_result_type<_Rp (*)(_A1)>
    : public unary_function<_A1, _Rp>
{
};

template <class _Rp, class _Cp>
struct __weak_result_type<_Rp (_Cp::*)()>
    : public unary_function<_Cp*, _Rp>
{
};

template <class _Rp, class _Cp>
struct __weak_result_type<_Rp (_Cp::*)() const>
    : public unary_function<const _Cp*, _Rp>
{
};

template <class _Rp, class _Cp>
struct __weak_result_type<_Rp (_Cp::*)() volatile>
    : public unary_function<volatile _Cp*, _Rp>
{
};

template <class _Rp, class _Cp>
struct __weak_result_type<_Rp (_Cp::*)() const volatile>
    : public unary_function<const volatile _Cp*, _Rp>
{
};

// 2 argument case

template <class _Rp, class _A1, class _A2>
struct __weak_result_type<_Rp (_A1, _A2)>
    : public binary_function<_A1, _A2, _Rp>
{
};

template <class _Rp, class _A1, class _A2>
struct __weak_result_type<_Rp (*)(_A1, _A2)>
    : public binary_function<_A1, _A2, _Rp>
{
};

template <class _Rp, class _A1, class _A2>
struct __weak_result_type<_Rp (&)(_A1, _A2)>
    : public binary_function<_A1, _A2, _Rp>
{
};

template <class _Rp, class _Cp, class _A1>
struct __weak_result_type<_Rp (_Cp::*)(_A1)>
    : public binary_function<_Cp*, _A1, _Rp>
{
};

template <class _Rp, class _Cp, class _A1>
struct __weak_result_type<_Rp (_Cp::*)(_A1) const>
    : public binary_function<const _Cp*, _A1, _Rp>
{
};

template <class _Rp, class _Cp, class _A1>
struct __weak_result_type<_Rp (_Cp::*)(_A1) volatile>
    : public binary_function<volatile _Cp*, _A1, _Rp>
{
};

template <class _Rp, class _Cp, class _A1>
struct __weak_result_type<_Rp (_Cp::*)(_A1) const volatile>
    : public binary_function<const volatile _Cp*, _A1, _Rp>
{
};

// 3 or more arguments

template <class _Rp, class _A1, class _A2, class _A3, class ..._A4>
struct __weak_result_type<_Rp (_A1, _A2, _A3, _A4...)>
{
    typedef _Rp result_type;
};

template <class _Rp, class _A1, class _A2, class _A3, class ..._A4>
struct __weak_result_type<_Rp (&)(_A1, _A2, _A3, _A4...)>
{
    typedef _Rp result_type;
};

template <class _Rp, class _A1, class _A2, class _A3, class ..._A4>
struct __weak_result_type<_Rp (*)(_A1, _A2, _A3, _A4...)>
{
    typedef _Rp result_type;
};

template <class _Rp, class _Cp, class _A1, class _A2, class ..._A3>
struct __weak_result_type<_Rp (_Cp::*)(_A1, _A2, _A3...)>
{
    typedef _Rp result_type;
};

template <class _Rp, class _Cp, class _A1, class _A2, class ..._A3>
struct __weak_result_type<_Rp (_Cp::*)(_A1, _A2, _A3...) const>
{
    typedef _Rp result_type;
};

template <class _Rp, class _Cp, class _A1, class _A2, class ..._A3>
struct __weak_result_type<_Rp (_Cp::*)(_A1, _A2, _A3...) volatile>
{
    typedef _Rp result_type;
};

template <class _Rp, class _Cp, class _A1, class _A2, class ..._A3>
struct __weak_result_type<_Rp (_Cp::*)(_A1, _A2, _A3...) const volatile>
{
    typedef _Rp result_type;
};

template <class _Tp, class ..._Args>
struct __invoke_return
{
    typedef decltype(_VSTD::__invoke(declval<_Tp>(), declval<_Args>()...)) type;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FUNCTIONAL_WEAK_RESULT_TYPE_H
