// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -emit-module-interface \
// RUN:     -fprebuilt-module-path=%t -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t \
// RUN:     -fsyntax-only -verify

//--- foo.h
#ifndef FOO_H
#define FOO_H

template<class _Tp>
concept __has_member_value_type = requires { typename _Tp::value_type; };

template<class _Tp>
concept __has_member_element_type = requires { typename _Tp::element_type; };

template <class _Tp>
inline constexpr bool is_object_v = __is_object(_Tp);

template<class> struct __cond_value_type {};

template<class _Tp>
requires is_object_v<_Tp>
struct __cond_value_type<_Tp> { using value_type = bool; };

template<class> struct indirectly_readable_traits {
    static constexpr int value = false;
};
#endif

//--- foo.member_value_type.h
#include "foo.h"
template<__has_member_value_type _Tp>
struct indirectly_readable_traits<_Tp> : __cond_value_type<typename _Tp::value_type> {
    static constexpr int value = false;
};

//--- foo.memeber_element_type.h
#include "foo.h"
template<__has_member_element_type _Tp>
struct indirectly_readable_traits<_Tp>  : __cond_value_type<typename _Tp::element_type>  {
    static constexpr int value = false;
};

template<__has_member_value_type _Tp>
  requires __has_member_element_type<_Tp>
struct indirectly_readable_traits<_Tp> {
    static constexpr int value = true;
};

//--- foo.a.h
#include "foo.h"
#include "foo.member_value_type.h"
#include "foo.memeber_element_type.h"
template <typename T>
using AType  = indirectly_readable_traits<T>;

//--- a.cppm
module;
#include "foo.a.h"
export module a;

export using ::AType;

//--- b.cppm
module;
#include "foo.h"
#include "foo.memeber_element_type.h"
export module b;

//--- c.cppm
export module c;

export import a;
export import b;

//--- use.cpp
// expected-no-diagnostics
import c;

template <typename T>
class U {
public:
    using value_type = T;
    using element_type = T;
};

template <typename T>
class V {
public:
};

static_assert(!AType<V<int*>>::value);
static_assert(AType<U<int**>>::value);
