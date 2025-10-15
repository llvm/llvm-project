//===--- MockHeaders.cpp - Mock headers for dataflow analyses -*- C++ ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines mock headers for testing of dataflow analyses.
//
//===----------------------------------------------------------------------===//

#include "MockHeaders.h"

namespace clang {
namespace dataflow {
namespace test {
static constexpr char CStdDefHeader[] = R"(
#ifndef CSTDDEF_H
#define CSTDDEF_H

namespace std {

typedef decltype(sizeof(char)) size_t;

using nullptr_t = decltype(nullptr);

} // namespace std

typedef decltype(sizeof(char)) size_t;
typedef decltype(sizeof(char*)) ptrdiff_t;

#endif // CSTDDEF_H
)";

static constexpr char StdTypeTraitsHeader[] = R"(
#ifndef STD_TYPE_TRAITS_H
#define STD_TYPE_TRAITS_H

#include "cstddef.h"

namespace std {

template <typename T, T V>
struct integral_constant {
  static constexpr T value = V;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template< class T > struct remove_reference      {typedef T type;};
template< class T > struct remove_reference<T&>  {typedef T type;};
template< class T > struct remove_reference<T&&> {typedef T type;};

template <class T>
  using remove_reference_t = typename remove_reference<T>::type;

template <class T>
struct remove_extent {
  typedef T type;
};

template <class T>
struct remove_extent<T[]> {
  typedef T type;
};

template <class T, size_t N>
struct remove_extent<T[N]> {
  typedef T type;
};

template <class T>
struct is_array : false_type {};

template <class T>
struct is_array<T[]> : true_type {};

template <class T, size_t N>
struct is_array<T[N]> : true_type {};

template <class>
struct is_function : false_type {};

template <class Ret, class... Args>
struct is_function<Ret(Args...)> : true_type {};

namespace detail {

template <class T>
struct type_identity {
  using type = T;
};  // or use type_identity (since C++20)

template <class T>
auto try_add_pointer(int) -> type_identity<typename remove_reference<T>::type*>;
template <class T>
auto try_add_pointer(...) -> type_identity<T>;

}  // namespace detail

template <class T>
struct add_pointer : decltype(detail::try_add_pointer<T>(0)) {};

template <bool B, class T, class F>
struct conditional {
  typedef T type;
};

template <class T, class F>
struct conditional<false, T, F> {
  typedef F type;
};

template <class T>
struct remove_cv {
  typedef T type;
};
template <class T>
struct remove_cv<const T> {
  typedef T type;
};
template <class T>
struct remove_cv<volatile T> {
  typedef T type;
};
template <class T>
struct remove_cv<const volatile T> {
  typedef T type;
};

template <class T>
using remove_cv_t = typename remove_cv<T>::type;

template <class T>
struct decay {
 private:
  typedef typename remove_reference<T>::type U;

 public:
  typedef typename conditional<
      is_array<U>::value, typename remove_extent<U>::type*,
      typename conditional<is_function<U>::value, typename add_pointer<U>::type,
                           typename remove_cv<U>::type>::type>::type type;
};

template <bool B, class T = void>
struct enable_if {};

template <class T>
struct enable_if<true, T> {
  typedef T type;
};

template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <class T, class U>
struct is_same : false_type {};

template <class T>
struct is_same<T, T> : true_type {};

template <class T>
struct is_void : is_same<void, typename remove_cv<T>::type> {};

namespace detail {

template <class T>
auto try_add_lvalue_reference(int) -> type_identity<T&>;
template <class T>
auto try_add_lvalue_reference(...) -> type_identity<T>;

template <class T>
auto try_add_rvalue_reference(int) -> type_identity<T&&>;
template <class T>
auto try_add_rvalue_reference(...) -> type_identity<T>;

}  // namespace detail

template <class T>
struct add_lvalue_reference : decltype(detail::try_add_lvalue_reference<T>(0)) {
};

template <class T>
struct add_rvalue_reference : decltype(detail::try_add_rvalue_reference<T>(0)) {
};

template <class T>
typename add_rvalue_reference<T>::type declval() noexcept;

namespace detail {

template <class T>
auto test_returnable(int)
    -> decltype(void(static_cast<T (*)()>(nullptr)), true_type{});
template <class>
auto test_returnable(...) -> false_type;

template <class From, class To>
auto test_implicitly_convertible(int)
    -> decltype(void(declval<void (&)(To)>()(declval<From>())), true_type{});
template <class, class>
auto test_implicitly_convertible(...) -> false_type;

}  // namespace detail

template <class From, class To>
struct is_convertible
    : integral_constant<bool,
                        (decltype(detail::test_returnable<To>(0))::value &&
                         decltype(detail::test_implicitly_convertible<From, To>(
                             0))::value) ||
                            (is_void<From>::value && is_void<To>::value)> {};

template <class From, class To>
inline constexpr bool is_convertible_v = is_convertible<From, To>::value;

template <class...>
using void_t = void;

template <class, class T, class... Args>
struct is_constructible_ : false_type {};

template <class T, class... Args>
struct is_constructible_<void_t<decltype(T(declval<Args>()...))>, T, Args...>
    : true_type {};

template <class T, class... Args>
using is_constructible = is_constructible_<void_t<>, T, Args...>;

template <class T, class... Args>
inline constexpr bool is_constructible_v = is_constructible<T, Args...>::value;

template <class _Tp>
struct __uncvref {
  typedef typename remove_cv<typename remove_reference<_Tp>::type>::type type;
};

template <class _Tp>
using __uncvref_t = typename __uncvref<_Tp>::type;

template <bool _Val>
using _BoolConstant = integral_constant<bool, _Val>;

template <class _Tp, class _Up>
using _IsSame = _BoolConstant<__is_same(_Tp, _Up)>;

template <class _Tp, class _Up>
using _IsNotSame = _BoolConstant<!__is_same(_Tp, _Up)>;

template <bool>
struct _MetaBase;
template <>
struct _MetaBase<true> {
  template <class _Tp, class _Up>
  using _SelectImpl = _Tp;
  template <template <class...> class _FirstFn, template <class...> class,
            class... _Args>
  using _SelectApplyImpl = _FirstFn<_Args...>;
  template <class _First, class...>
  using _FirstImpl = _First;
  template <class, class _Second, class...>
  using _SecondImpl = _Second;
  template <class _Result, class _First, class... _Rest>
  using _OrImpl =
      typename _MetaBase<_First::value != true && sizeof...(_Rest) != 0>::
          template _OrImpl<_First, _Rest...>;
};

template <>
struct _MetaBase<false> {
  template <class _Tp, class _Up>
  using _SelectImpl = _Up;
  template <template <class...> class, template <class...> class _SecondFn,
            class... _Args>
  using _SelectApplyImpl = _SecondFn<_Args...>;
  template <class _Result, class...>
  using _OrImpl = _Result;
};

template <bool _Cond, class _IfRes, class _ElseRes>
using _If = typename _MetaBase<_Cond>::template _SelectImpl<_IfRes, _ElseRes>;

template <class... _Rest>
using _Or = typename _MetaBase<sizeof...(_Rest) !=
                               0>::template _OrImpl<false_type, _Rest...>;

template <bool _Bp, class _Tp = void>
using __enable_if_t = typename enable_if<_Bp, _Tp>::type;

template <class...>
using __expand_to_true = true_type;
template <class... _Pred>
__expand_to_true<__enable_if_t<_Pred::value>...> __and_helper(int);
template <class...>
false_type __and_helper(...);
template <class... _Pred>
using _And = decltype(__and_helper<_Pred...>(0));

template <class _Pred>
struct _Not : _BoolConstant<!_Pred::value> {};

struct __check_tuple_constructor_fail {
  static constexpr bool __enable_explicit_default() { return false; }
  static constexpr bool __enable_implicit_default() { return false; }
  template <class...>
  static constexpr bool __enable_explicit() {
    return false;
  }
  template <class...>
  static constexpr bool __enable_implicit() {
    return false;
  }
};

template <typename, typename _Tp>
struct __select_2nd {
  typedef _Tp type;
};
template <class _Tp, class _Arg>
typename __select_2nd<decltype((declval<_Tp>() = declval<_Arg>())),
                      true_type>::type
__is_assignable_test(int);
template <class, class>
false_type __is_assignable_test(...);
template <class _Tp, class _Arg,
          bool = is_void<_Tp>::value || is_void<_Arg>::value>
struct __is_assignable_imp
    : public decltype((__is_assignable_test<_Tp, _Arg>(0))) {};
template <class _Tp, class _Arg>
struct __is_assignable_imp<_Tp, _Arg, true> : public false_type {};
template <class _Tp, class _Arg>
struct is_assignable : public __is_assignable_imp<_Tp, _Arg> {};

template <class _Tp>
struct __libcpp_is_integral : public false_type {};
template <>
struct __libcpp_is_integral<bool> : public true_type {};
template <>
struct __libcpp_is_integral<char> : public true_type {};
template <>
struct __libcpp_is_integral<signed char> : public true_type {};
template <>
struct __libcpp_is_integral<unsigned char> : public true_type {};
template <>
struct __libcpp_is_integral<wchar_t> : public true_type {};
template <>
struct __libcpp_is_integral<short> : public true_type {};  // NOLINT
template <>
struct __libcpp_is_integral<unsigned short> : public true_type {};  // NOLINT
template <>
struct __libcpp_is_integral<int> : public true_type {};
template <>
struct __libcpp_is_integral<unsigned int> : public true_type {};
template <>
struct __libcpp_is_integral<long> : public true_type {};  // NOLINT
template <>
struct __libcpp_is_integral<unsigned long> : public true_type {};  // NOLINT
template <>
struct __libcpp_is_integral<long long> : public true_type {};  // NOLINT
template <>                                                    // NOLINTNEXTLINE
struct __libcpp_is_integral<unsigned long long> : public true_type {};
template <class _Tp>
struct is_integral
    : public __libcpp_is_integral<typename remove_cv<_Tp>::type> {};

template <class _Tp>
struct __libcpp_is_floating_point : public false_type {};
template <>
struct __libcpp_is_floating_point<float> : public true_type {};
template <>
struct __libcpp_is_floating_point<double> : public true_type {};
template <>
struct __libcpp_is_floating_point<long double> : public true_type {};
template <class _Tp>
struct is_floating_point
    : public __libcpp_is_floating_point<typename remove_cv<_Tp>::type> {};

template <class _Tp>
struct is_arithmetic
    : public integral_constant<bool, is_integral<_Tp>::value ||
                                         is_floating_point<_Tp>::value> {};

template <class _Tp>
struct __libcpp_is_pointer : public false_type {};
template <class _Tp>
struct __libcpp_is_pointer<_Tp*> : public true_type {};
template <class _Tp>
struct is_pointer : public __libcpp_is_pointer<typename remove_cv<_Tp>::type> {
};

template <class _Tp>
struct __libcpp_is_member_pointer : public false_type {};
template <class _Tp, class _Up>
struct __libcpp_is_member_pointer<_Tp _Up::*> : public true_type {};
template <class _Tp>
struct is_member_pointer
    : public __libcpp_is_member_pointer<typename remove_cv<_Tp>::type> {};

template <class _Tp>
struct __libcpp_union : public false_type {};
template <class _Tp>
struct is_union : public __libcpp_union<typename remove_cv<_Tp>::type> {};

template <class T>
struct is_reference : false_type {};
template <class T>
struct is_reference<T&> : true_type {};
template <class T>
struct is_reference<T&&> : true_type {};

template <class T>
inline constexpr bool is_reference_v = is_reference<T>::value;

struct __two {
  char __lx[2];
};

namespace __is_class_imp {
template <class _Tp>
char __test(int _Tp::*);
template <class _Tp>
__two __test(...);
}  // namespace __is_class_imp
template <class _Tp>
struct is_class
    : public integral_constant<bool,
                               sizeof(__is_class_imp::__test<_Tp>(0)) == 1 &&
                                   !is_union<_Tp>::value> {};

template <class _Tp>
struct __is_nullptr_t_impl : public false_type {};
template <>
struct __is_nullptr_t_impl<nullptr_t> : public true_type {};
template <class _Tp>
struct __is_nullptr_t
    : public __is_nullptr_t_impl<typename remove_cv<_Tp>::type> {};
template <class _Tp>
struct is_null_pointer
    : public __is_nullptr_t_impl<typename remove_cv<_Tp>::type> {};

template <class _Tp>
struct is_enum
    : public integral_constant<
          bool, !is_void<_Tp>::value && !is_integral<_Tp>::value &&
                    !is_floating_point<_Tp>::value && !is_array<_Tp>::value &&
                    !is_pointer<_Tp>::value && !is_reference<_Tp>::value &&
                    !is_member_pointer<_Tp>::value && !is_union<_Tp>::value &&
                    !is_class<_Tp>::value && !is_function<_Tp>::value> {};

template <class _Tp>
struct is_scalar
    : public integral_constant<
          bool, is_arithmetic<_Tp>::value || is_member_pointer<_Tp>::value ||
                    is_pointer<_Tp>::value || __is_nullptr_t<_Tp>::value ||
                    is_enum<_Tp>::value> {};
template <>
struct is_scalar<nullptr_t> : public true_type {};

} // namespace std

#endif // STD_TYPE_TRAITS_H
)";

static constexpr char AbslTypeTraitsHeader[] = R"(
#ifndef ABSL_TYPE_TRAITS_H
#define ABSL_TYPE_TRAITS_H

#include "std_type_traits.h"

namespace absl {

template <typename... Ts>
struct conjunction : std::true_type {};

template <typename T, typename... Ts>
struct conjunction<T, Ts...>
    : std::conditional<T::value, conjunction<Ts...>, T>::type {};

template <typename T>
struct conjunction<T> : T {};

template <typename... Ts>
struct disjunction : std::false_type {};

template <typename T, typename... Ts>
struct disjunction<T, Ts...>
    : std::conditional<T::value, T, disjunction<Ts...>>::type {};

template <typename T>
struct disjunction<T> : T {};

template <typename T>
struct negation : std::integral_constant<bool, !T::value> {};

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;


template <bool B, typename T, typename F>
using conditional_t = typename std::conditional<B, T, F>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using decay_t = typename std::decay<T>::type;

struct in_place_t {};

constexpr in_place_t in_place;
} // namespace absl

#endif // ABSL_TYPE_TRAITS_H
)";

static constexpr char StdStringHeader[] = R"(
#ifndef STRING_H
#define STRING_H

namespace std {

struct string {
  string(const char*);
  ~string();
  const char *c_str() const;
  bool empty();
};

struct string_view {
  string_view(const char*);
  ~string_view();
  bool empty();
};

bool operator!=(const string &LHS, const char *RHS);

} // namespace std

#endif // STRING_H
)";

static constexpr char StdUtilityHeader[] = R"(
#ifndef UTILITY_H
#define UTILITY_H

#include "std_type_traits.h"

namespace std {

template <typename T>
constexpr remove_reference_t<T>&& move(T&& x);

template <typename T>
void swap(T& a, T& b) noexcept;

} // namespace std

#endif // UTILITY_H
)";

static constexpr char StdInitializerListHeader[] = R"(
#ifndef INITIALIZER_LIST_H
#define INITIALIZER_LIST_H

namespace std {

template <typename T>
class initializer_list {
 public:
  const T *a, *b;
  initializer_list() noexcept;
};

} // namespace std

#endif // INITIALIZER_LIST_H
)";

static constexpr char StdOptionalHeader[] = R"(
#include "std_initializer_list.h"
#include "std_type_traits.h"
#include "std_utility.h"

namespace std {

struct in_place_t {};
constexpr in_place_t in_place;

struct nullopt_t {
  constexpr explicit nullopt_t() {}
};
constexpr nullopt_t nullopt;

template <class _Tp>
struct __optional_destruct_base {
  constexpr void reset() noexcept;
};

template <class _Tp>
struct __optional_storage_base : __optional_destruct_base<_Tp> {
  constexpr bool has_value() const noexcept;
};

template <typename _Tp>
class optional : private __optional_storage_base<_Tp> {
  using __base = __optional_storage_base<_Tp>;

 public:
  using value_type = _Tp;

 private:
  struct _CheckOptionalArgsConstructor {
    template <class _Up>
    static constexpr bool __enable_implicit() {
      return is_constructible_v<_Tp, _Up&&> && is_convertible_v<_Up&&, _Tp>;
    }

    template <class _Up>
    static constexpr bool __enable_explicit() {
      return is_constructible_v<_Tp, _Up&&> && !is_convertible_v<_Up&&, _Tp>;
    }
  };
  template <class _Up>
  using _CheckOptionalArgsCtor =
      _If<_IsNotSame<__uncvref_t<_Up>, in_place_t>::value &&
              _IsNotSame<__uncvref_t<_Up>, optional>::value,
          _CheckOptionalArgsConstructor, __check_tuple_constructor_fail>;
  template <class _QualUp>
  struct _CheckOptionalLikeConstructor {
    template <class _Up, class _Opt = optional<_Up>>
    using __check_constructible_from_opt =
        _Or<is_constructible<_Tp, _Opt&>, is_constructible<_Tp, _Opt const&>,
            is_constructible<_Tp, _Opt&&>, is_constructible<_Tp, _Opt const&&>,
            is_convertible<_Opt&, _Tp>, is_convertible<_Opt const&, _Tp>,
            is_convertible<_Opt&&, _Tp>, is_convertible<_Opt const&&, _Tp>>;
    template <class _Up, class _QUp = _QualUp>
    static constexpr bool __enable_implicit() {
      return is_convertible<_QUp, _Tp>::value &&
             !__check_constructible_from_opt<_Up>::value;
    }
    template <class _Up, class _QUp = _QualUp>
    static constexpr bool __enable_explicit() {
      return !is_convertible<_QUp, _Tp>::value &&
             !__check_constructible_from_opt<_Up>::value;
    }
  };

  template <class _Up, class _QualUp>
  using _CheckOptionalLikeCtor =
      _If<_And<_IsNotSame<_Up, _Tp>, is_constructible<_Tp, _QualUp>>::value,
          _CheckOptionalLikeConstructor<_QualUp>,
          __check_tuple_constructor_fail>;


  template <class _Up, class _QualUp>
  using _CheckOptionalLikeAssign = _If<
      _And<
          _IsNotSame<_Up, _Tp>,
          is_constructible<_Tp, _QualUp>,
          is_assignable<_Tp&, _QualUp>
      >::value,
      _CheckOptionalLikeConstructor<_QualUp>,
      __check_tuple_constructor_fail
    >;

 public:
  constexpr optional() noexcept {}
  constexpr optional(const optional&) = default;
  constexpr optional(optional&&) = default;
  constexpr optional(nullopt_t) noexcept {}

  template <
      class _InPlaceT, class... _Args,
      class = enable_if_t<_And<_IsSame<_InPlaceT, in_place_t>,
                             is_constructible<value_type, _Args...>>::value>>
  constexpr explicit optional(_InPlaceT, _Args&&... __args);

  template <class _Up, class... _Args,
            class = enable_if_t<is_constructible_v<
                value_type, initializer_list<_Up>&, _Args...>>>
  constexpr explicit optional(in_place_t, initializer_list<_Up> __il,
                              _Args&&... __args);

  template <
      class _Up = value_type,
      enable_if_t<_CheckOptionalArgsCtor<_Up>::template __enable_implicit<_Up>(),
                int> = 0>
  constexpr optional(_Up&& __v);

  template <
      class _Up,
      enable_if_t<_CheckOptionalArgsCtor<_Up>::template __enable_explicit<_Up>(),
                int> = 0>
  constexpr explicit optional(_Up&& __v);

  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up const&>::
                                     template __enable_implicit<_Up>(),
                                 int> = 0>
  constexpr optional(const optional<_Up>& __v);

  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up const&>::
                                     template __enable_explicit<_Up>(),
                                 int> = 0>
  constexpr explicit optional(const optional<_Up>& __v);

  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up&&>::
                                     template __enable_implicit<_Up>(),
                                 int> = 0>
  constexpr optional(optional<_Up>&& __v);

  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up&&>::
                                     template __enable_explicit<_Up>(),
                                 int> = 0>
  constexpr explicit optional(optional<_Up>&& __v);

  constexpr optional& operator=(nullopt_t) noexcept;

  optional& operator=(const optional&);

  optional& operator=(optional&&);

  template <class _Up = value_type,
            class = enable_if_t<_And<_IsNotSame<__uncvref_t<_Up>, optional>,
                                   _Or<_IsNotSame<__uncvref_t<_Up>, value_type>,
                                       _Not<is_scalar<value_type>>>,
                                   is_constructible<value_type, _Up>,
                                   is_assignable<value_type&, _Up>>::value>>
  constexpr optional& operator=(_Up&& __v);

  template <class _Up, enable_if_t<_CheckOptionalLikeAssign<_Up, _Up const&>::
                                     template __enable_assign<_Up>(),
                                 int> = 0>
  constexpr optional& operator=(const optional<_Up>& __v);

  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up&&>::
                                     template __enable_assign<_Up>(),
                                 int> = 0>
  constexpr optional& operator=(optional<_Up>&& __v);

  const _Tp& operator*() const&;
  _Tp& operator*() &;
  const _Tp&& operator*() const&&;
  _Tp&& operator*() &&;

  const _Tp* operator->() const;
  _Tp* operator->();

  const _Tp& value() const&;
  _Tp& value() &;
  const _Tp&& value() const&&;
  _Tp&& value() &&;

  template <typename U>
  constexpr _Tp value_or(U&& v) const&;
  template <typename U>
  _Tp value_or(U&& v) &&;

  template <typename... Args>
  _Tp& emplace(Args&&... args);

  template <typename U, typename... Args>
  _Tp& emplace(std::initializer_list<U> ilist, Args&&... args);

  using __base::reset;

  constexpr explicit operator bool() const noexcept;
  using __base::has_value;

  constexpr void swap(optional& __opt) noexcept;
};

template <typename T>
constexpr optional<typename std::decay<T>::type> make_optional(T&& v);

template <typename T, typename... Args>
constexpr optional<T> make_optional(Args&&... args);

template <typename T, typename U, typename... Args>
constexpr optional<T> make_optional(std::initializer_list<U> il,
                                    Args&&... args);

template <typename T, typename U>
constexpr bool operator==(const optional<T> &lhs, const optional<U> &rhs);
template <typename T, typename U>
constexpr bool operator!=(const optional<T> &lhs, const optional<U> &rhs);

template <typename T>
constexpr bool operator==(const optional<T> &opt, nullopt_t);

// C++20 and later do not define the following overloads because they are
// provided by rewritten candidates instead.
#if __cplusplus < 202002L
template <typename T>
constexpr bool operator==(nullopt_t, const optional<T> &opt);
template <typename T>
constexpr bool operator!=(const optional<T> &opt, nullopt_t);
template <typename T>
constexpr bool operator!=(nullopt_t, const optional<T> &opt);
#endif  // __cplusplus < 202002L

template <typename T, typename U>
constexpr bool operator==(const optional<T> &opt, const U &value);
template <typename T, typename U>
constexpr bool operator==(const T &value, const optional<U> &opt);
template <typename T, typename U>
constexpr bool operator!=(const optional<T> &opt, const U &value);
template <typename T, typename U>
constexpr bool operator!=(const T &value, const optional<U> &opt);

} // namespace std
)";

static constexpr char AbslOptionalHeader[] = R"(
#include "absl_type_traits.h"
#include "std_initializer_list.h"
#include "std_type_traits.h"
#include "std_utility.h"

namespace absl {

struct nullopt_t {
  constexpr explicit nullopt_t() {}
};
constexpr nullopt_t nullopt;

template <typename T>
class optional;

namespace optional_internal {

template <typename T, typename U>
struct is_constructible_convertible_from_optional
    : std::integral_constant<
          bool, std::is_constructible<T, optional<U>&>::value ||
                    std::is_constructible<T, optional<U>&&>::value ||
                    std::is_constructible<T, const optional<U>&>::value ||
                    std::is_constructible<T, const optional<U>&&>::value ||
                    std::is_convertible<optional<U>&, T>::value ||
                    std::is_convertible<optional<U>&&, T>::value ||
                    std::is_convertible<const optional<U>&, T>::value ||
                    std::is_convertible<const optional<U>&&, T>::value> {};

template <typename T, typename U>
struct is_constructible_convertible_assignable_from_optional
    : std::integral_constant<
          bool, is_constructible_convertible_from_optional<T, U>::value ||
                    std::is_assignable<T&, optional<U>&>::value ||
                    std::is_assignable<T&, optional<U>&&>::value ||
                    std::is_assignable<T&, const optional<U>&>::value ||
                    std::is_assignable<T&, const optional<U>&&>::value> {};

}  // namespace optional_internal

template <typename T>
class optional {
 public:
  constexpr optional() noexcept;

  constexpr optional(nullopt_t) noexcept;

  optional(const optional&) = default;

  optional(optional&&) = default;

  template <typename InPlaceT, typename... Args,
            absl::enable_if_t<absl::conjunction<
                std::is_same<InPlaceT, in_place_t>,
                std::is_constructible<T, Args&&...>>::value>* = nullptr>
  constexpr explicit optional(InPlaceT, Args&&... args);

  template <typename U, typename... Args,
            typename = typename std::enable_if<std::is_constructible<
                T, std::initializer_list<U>&, Args&&...>::value>::type>
  constexpr explicit optional(in_place_t, std::initializer_list<U> il,
                              Args&&... args);

  template <
      typename U = T,
      typename std::enable_if<
          absl::conjunction<absl::negation<std::is_same<
                                in_place_t, typename std::decay<U>::type>>,
                            absl::negation<std::is_same<
                                optional<T>, typename std::decay<U>::type>>,
                            std::is_convertible<U&&, T>,
                            std::is_constructible<T, U&&>>::value,
          bool>::type = false>
  constexpr optional(U&& v);

  template <
      typename U = T,
      typename std::enable_if<
          absl::conjunction<absl::negation<std::is_same<
                                in_place_t, typename std::decay<U>::type>>,
                            absl::negation<std::is_same<
                                optional<T>, typename std::decay<U>::type>>,
                            absl::negation<std::is_convertible<U&&, T>>,
                            std::is_constructible<T, U&&>>::value,
          bool>::type = false>
  explicit constexpr optional(U&& v);

  template <typename U,
            typename std::enable_if<
                absl::conjunction<
                    absl::negation<std::is_same<T, U>>,
                    std::is_constructible<T, const U&>,
                    absl::negation<
                        optional_internal::
                            is_constructible_convertible_from_optional<T, U>>,
                    std::is_convertible<const U&, T>>::value,
                bool>::type = false>
  optional(const optional<U>& rhs);

  template <typename U,
            typename std::enable_if<
                absl::conjunction<
                    absl::negation<std::is_same<T, U>>,
                    std::is_constructible<T, const U&>,
                    absl::negation<
                        optional_internal::
                            is_constructible_convertible_from_optional<T, U>>,
                    absl::negation<std::is_convertible<const U&, T>>>::value,
                bool>::type = false>
  explicit optional(const optional<U>& rhs);

  template <
      typename U,
      typename std::enable_if<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>, std::is_constructible<T, U&&>,
              absl::negation<
                  optional_internal::is_constructible_convertible_from_optional<
                      T, U>>,
              std::is_convertible<U&&, T>>::value,
          bool>::type = false>
  optional(optional<U>&& rhs);

  template <
      typename U,
      typename std::enable_if<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>, std::is_constructible<T, U&&>,
              absl::negation<
                  optional_internal::is_constructible_convertible_from_optional<
                      T, U>>,
              absl::negation<std::is_convertible<U&&, T>>>::value,
          bool>::type = false>
  explicit optional(optional<U>&& rhs);

  optional& operator=(nullopt_t) noexcept;

  optional& operator=(const optional& src);

  optional& operator=(optional&& src);

  template <
      typename U = T,
      typename = typename std::enable_if<absl::conjunction<
          absl::negation<
              std::is_same<optional<T>, typename std::decay<U>::type>>,
          absl::negation<
              absl::conjunction<std::is_scalar<T>,
                                std::is_same<T, typename std::decay<U>::type>>>,
          std::is_constructible<T, U>, std::is_assignable<T&, U>>::value>::type>
  optional& operator=(U&& v);

  template <
      typename U,
      typename = typename std::enable_if<absl::conjunction<
          absl::negation<std::is_same<T, U>>,
          std::is_constructible<T, const U&>, std::is_assignable<T&, const U&>,
          absl::negation<
              optional_internal::
                  is_constructible_convertible_assignable_from_optional<
                      T, U>>>::value>::type>
  optional& operator=(const optional<U>& rhs);

  template <typename U,
            typename = typename std::enable_if<absl::conjunction<
                absl::negation<std::is_same<T, U>>, std::is_constructible<T, U>,
                std::is_assignable<T&, U>,
                absl::negation<
                    optional_internal::
                        is_constructible_convertible_assignable_from_optional<
                            T, U>>>::value>::type>
  optional& operator=(optional<U>&& rhs);

  const T& operator*() const&;
  T& operator*() &;
  const T&& operator*() const&&;
  T&& operator*() &&;

  const T* operator->() const;
  T* operator->();

  const T& value() const&;
  T& value() &;
  const T&& value() const&&;
  T&& value() &&;

  template <typename U>
  constexpr T value_or(U&& v) const&;
  template <typename U>
  T value_or(U&& v) &&;

  template <typename... Args>
  T& emplace(Args&&... args);

  template <typename U, typename... Args>
  T& emplace(std::initializer_list<U> ilist, Args&&... args);

  void reset() noexcept;

  constexpr explicit operator bool() const noexcept;
  constexpr bool has_value() const noexcept;

  void swap(optional& rhs) noexcept;
};

template <typename T>
constexpr optional<typename std::decay<T>::type> make_optional(T&& v);

template <typename T, typename... Args>
constexpr optional<T> make_optional(Args&&... args);

template <typename T, typename U, typename... Args>
constexpr optional<T> make_optional(std::initializer_list<U> il,
                                    Args&&... args);

template <typename T, typename U>
constexpr bool operator==(const optional<T> &lhs, const optional<U> &rhs);
template <typename T, typename U>
constexpr bool operator!=(const optional<T> &lhs, const optional<U> &rhs);

template <typename T>
constexpr bool operator==(const optional<T> &opt, nullopt_t);
template <typename T>
constexpr bool operator==(nullopt_t, const optional<T> &opt);
template <typename T>
constexpr bool operator!=(const optional<T> &opt, nullopt_t);
template <typename T>
constexpr bool operator!=(nullopt_t, const optional<T> &opt);

template <typename T, typename U>
constexpr bool operator==(const optional<T> &opt, const U &value);
template <typename T, typename U>
constexpr bool operator==(const T &value, const optional<U> &opt);
template <typename T, typename U>
constexpr bool operator!=(const optional<T> &opt, const U &value);
template <typename T, typename U>
constexpr bool operator!=(const T &value, const optional<U> &opt);

} // namespace absl
)";

static constexpr char BaseOptionalHeader[] = R"(
#include "std_initializer_list.h"
#include "std_type_traits.h"
#include "std_utility.h"

namespace base {

struct in_place_t {};
constexpr in_place_t in_place;

struct nullopt_t {
  constexpr explicit nullopt_t() {}
};
constexpr nullopt_t nullopt;

template <typename T>
class Optional;

namespace internal {

template <typename T>
using RemoveCvRefT = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T, typename U>
struct IsConvertibleFromOptional
    : std::integral_constant<
          bool, std::is_constructible<T, Optional<U>&>::value ||
                    std::is_constructible<T, const Optional<U>&>::value ||
                    std::is_constructible<T, Optional<U>&&>::value ||
                    std::is_constructible<T, const Optional<U>&&>::value ||
                    std::is_convertible<Optional<U>&, T>::value ||
                    std::is_convertible<const Optional<U>&, T>::value ||
                    std::is_convertible<Optional<U>&&, T>::value ||
                    std::is_convertible<const Optional<U>&&, T>::value> {};

template <typename T, typename U>
struct IsAssignableFromOptional
    : std::integral_constant<
          bool, IsConvertibleFromOptional<T, U>::value ||
                    std::is_assignable<T&, Optional<U>&>::value ||
                    std::is_assignable<T&, const Optional<U>&>::value ||
                    std::is_assignable<T&, Optional<U>&&>::value ||
                    std::is_assignable<T&, const Optional<U>&&>::value> {};

}  // namespace internal

template <typename T>
class Optional {
 public:
  using value_type = T;

  constexpr Optional() = default;
  constexpr Optional(const Optional& other) noexcept = default;
  constexpr Optional(Optional&& other) noexcept = default;

  constexpr Optional(nullopt_t);

  template <typename U,
            typename std::enable_if<
                std::is_constructible<T, const U&>::value &&
                    !internal::IsConvertibleFromOptional<T, U>::value &&
                    std::is_convertible<const U&, T>::value,
                bool>::type = false>
  Optional(const Optional<U>& other) noexcept;

  template <typename U,
            typename std::enable_if<
                std::is_constructible<T, const U&>::value &&
                    !internal::IsConvertibleFromOptional<T, U>::value &&
                    !std::is_convertible<const U&, T>::value,
                bool>::type = false>
  explicit Optional(const Optional<U>& other) noexcept;

  template <typename U,
            typename std::enable_if<
                std::is_constructible<T, U&&>::value &&
                    !internal::IsConvertibleFromOptional<T, U>::value &&
                    std::is_convertible<U&&, T>::value,
                bool>::type = false>
  Optional(Optional<U>&& other) noexcept;

  template <typename U,
            typename std::enable_if<
                std::is_constructible<T, U&&>::value &&
                    !internal::IsConvertibleFromOptional<T, U>::value &&
                    !std::is_convertible<U&&, T>::value,
                bool>::type = false>
  explicit Optional(Optional<U>&& other) noexcept;

  template <class... Args>
  constexpr explicit Optional(in_place_t, Args&&... args);

  template <class U, class... Args,
            class = typename std::enable_if<std::is_constructible<
                value_type, std::initializer_list<U>&, Args...>::value>::type>
  constexpr explicit Optional(in_place_t, std::initializer_list<U> il,
                              Args&&... args);

  template <
      typename U = value_type,
      typename std::enable_if<
          std::is_constructible<T, U&&>::value &&
              !std::is_same<internal::RemoveCvRefT<U>, in_place_t>::value &&
              !std::is_same<internal::RemoveCvRefT<U>, Optional<T>>::value &&
              std::is_convertible<U&&, T>::value,
          bool>::type = false>
  constexpr Optional(U&& value);

  template <
      typename U = value_type,
      typename std::enable_if<
          std::is_constructible<T, U&&>::value &&
              !std::is_same<internal::RemoveCvRefT<U>, in_place_t>::value &&
              !std::is_same<internal::RemoveCvRefT<U>, Optional<T>>::value &&
              !std::is_convertible<U&&, T>::value,
          bool>::type = false>
  constexpr explicit Optional(U&& value);

  Optional& operator=(const Optional& other) noexcept;

  Optional& operator=(Optional&& other) noexcept;

  Optional& operator=(nullopt_t);

  template <typename U>
  typename std::enable_if<
      !std::is_same<internal::RemoveCvRefT<U>, Optional<T>>::value &&
          std::is_constructible<T, U>::value &&
          std::is_assignable<T&, U>::value &&
          (!std::is_scalar<T>::value ||
           !std::is_same<typename std::decay<U>::type, T>::value),
      Optional&>::type
  operator=(U&& value) noexcept;

  template <typename U>
  typename std::enable_if<!internal::IsAssignableFromOptional<T, U>::value &&
                              std::is_constructible<T, const U&>::value &&
                              std::is_assignable<T&, const U&>::value,
                          Optional&>::type
  operator=(const Optional<U>& other) noexcept;

  template <typename U>
  typename std::enable_if<!internal::IsAssignableFromOptional<T, U>::value &&
                              std::is_constructible<T, U>::value &&
                              std::is_assignable<T&, U>::value,
                          Optional&>::type
  operator=(Optional<U>&& other) noexcept;

  const T& operator*() const&;
  T& operator*() &;
  const T&& operator*() const&&;
  T&& operator*() &&;

  const T* operator->() const;
  T* operator->();

  const T& value() const&;
  T& value() &;
  const T&& value() const&&;
  T&& value() &&;

  template <typename U>
  constexpr T value_or(U&& v) const&;
  template <typename U>
  T value_or(U&& v) &&;

  template <typename... Args>
  T& emplace(Args&&... args);

  template <typename U, typename... Args>
  T& emplace(std::initializer_list<U> ilist, Args&&... args);

  void reset() noexcept;

  constexpr explicit operator bool() const noexcept;
  constexpr bool has_value() const noexcept;

  void swap(Optional& other);
};

template <typename T>
constexpr Optional<typename std::decay<T>::type> make_optional(T&& v);

template <typename T, typename... Args>
constexpr Optional<T> make_optional(Args&&... args);

template <typename T, typename U, typename... Args>
constexpr Optional<T> make_optional(std::initializer_list<U> il,
                                    Args&&... args);

template <typename T, typename U>
constexpr bool operator==(const Optional<T> &lhs, const Optional<U> &rhs);
template <typename T, typename U>
constexpr bool operator!=(const Optional<T> &lhs, const Optional<U> &rhs);

template <typename T>
constexpr bool operator==(const Optional<T> &opt, nullopt_t);
template <typename T>
constexpr bool operator==(nullopt_t, const Optional<T> &opt);
template <typename T>
constexpr bool operator!=(const Optional<T> &opt, nullopt_t);
template <typename T>
constexpr bool operator!=(nullopt_t, const Optional<T> &opt);

template <typename T, typename U>
constexpr bool operator==(const Optional<T> &opt, const U &value);
template <typename T, typename U>
constexpr bool operator==(const T &value, const Optional<U> &opt);
template <typename T, typename U>
constexpr bool operator!=(const Optional<T> &opt, const U &value);
template <typename T, typename U>
constexpr bool operator!=(const T &value, const Optional<U> &opt);

} // namespace base
)";

constexpr const char StatusDefsHeader[] =
    R"cc(
#ifndef STATUS_H_
#define STATUS_H_

#include "absl_type_traits.h"
#include "std_initializer_list.h"
#include "std_string.h"
#include "std_type_traits.h"
#include "std_utility.h"

namespace absl {
struct SourceLocation {
  static constexpr SourceLocation current();
  static constexpr SourceLocation
  DoNotInvokeDirectlyNoSeriouslyDont(int line, const char *file_name);
};
} // namespace absl
namespace absl {
enum class StatusCode : int {
  kOk,
  kCancelled,
  kUnknown,
  kInvalidArgument,
  kDeadlineExceeded,
  kNotFound,
  kAlreadyExists,
  kPermissionDenied,
  kResourceExhausted,
  kFailedPrecondition,
  kAborted,
  kOutOfRange,
  kUnimplemented,
  kInternal,
  kUnavailable,
  kDataLoss,
  kUnauthenticated,
};
} // namespace absl

namespace absl {
enum class StatusToStringMode : int {
  kWithNoExtraData = 0,
  kWithPayload = 1 << 0,
  kWithSourceLocation = 1 << 1,
  kWithEverything = ~kWithNoExtraData,
  kDefault = kWithPayload,
};
class Status {
public:
  Status();
  template <typename Enum> Status(Enum code, std::string_view msg);
  Status(absl::StatusCode code, std::string_view msg,
         absl::SourceLocation loc = SourceLocation::current());
  Status(const Status &base_status, absl::SourceLocation loc);
  Status(Status &&base_status, absl::SourceLocation loc);
  ~Status() {}

  Status(const Status &);
  Status &operator=(const Status &x);

  Status(Status &&) noexcept;
  Status &operator=(Status &&);

  friend bool operator==(const Status &, const Status &);
  friend bool operator!=(const Status &, const Status &);

  bool ok() const { return true; }
  void CheckSuccess() const;
  void IgnoreError() const;
  int error_code() const;
  absl::Status ToCanonical() const;
  std::string
  ToString(StatusToStringMode m = StatusToStringMode::kDefault) const;
  void Update(const Status &new_status);
  void Update(Status &&new_status);
};

bool operator==(const Status &lhs, const Status &rhs);
bool operator!=(const Status &lhs, const Status &rhs);

Status OkStatus();
Status InvalidArgumentError(char *);

#endif // STATUS_H
)cc";

constexpr const char StatusOrDefsHeader[] = R"cc(
#ifndef STATUSOR_H_
#define STATUSOR_H_
#include "absl_type_traits.h"
#include "status_defs.h"
#include "std_initializer_list.h"
#include "std_type_traits.h"
#include "std_utility.h"

template <typename T> struct StatusOr;

namespace internal_statusor {

template <typename T, typename U, typename = void>
struct HasConversionOperatorToStatusOr : std::false_type {};

template <typename T, typename U>
void test(char (*)[sizeof(std::declval<U>().operator absl::StatusOr<T>())]);

template <typename T, typename U>
struct HasConversionOperatorToStatusOr<T, U, decltype(test<T, U>(0))>
    : std::true_type {};

template <typename T, typename U>
using IsConstructibleOrConvertibleFromStatusOr =
    absl::disjunction<std::is_constructible<T, StatusOr<U> &>,
                      std::is_constructible<T, const StatusOr<U> &>,
                      std::is_constructible<T, StatusOr<U> &&>,
                      std::is_constructible<T, const StatusOr<U> &&>,
                      std::is_convertible<StatusOr<U> &, T>,
                      std::is_convertible<const StatusOr<U> &, T>,
                      std::is_convertible<StatusOr<U> &&, T>,
                      std::is_convertible<const StatusOr<U> &&, T>>;

template <typename T, typename U>
using IsConstructibleOrConvertibleOrAssignableFromStatusOr =
    absl::disjunction<IsConstructibleOrConvertibleFromStatusOr<T, U>,
                      std::is_assignable<T &, StatusOr<U> &>,
                      std::is_assignable<T &, const StatusOr<U> &>,
                      std::is_assignable<T &, StatusOr<U> &&>,
                      std::is_assignable<T &, const StatusOr<U> &&>>;

template <typename T, typename U>
struct IsDirectInitializationAmbiguous
    : public absl::conditional_t<
          std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                       U>::value,
          std::false_type,
          IsDirectInitializationAmbiguous<
              T, absl::remove_cv_t<absl::remove_reference_t<U>>>> {};

template <typename T, typename V>
struct IsDirectInitializationAmbiguous<T, absl::StatusOr<V>>
    : public IsConstructibleOrConvertibleFromStatusOr<T, V> {};

template <typename T, typename U>
using IsDirectInitializationValid = absl::disjunction<
    // Short circuits if T is basically U.
    std::is_same<T, absl::remove_cv_t<absl::remove_reference_t<U>>>,
    absl::negation<absl::disjunction<
        std::is_same<absl::StatusOr<T>,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::Status,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::in_place_t,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        IsDirectInitializationAmbiguous<T, U>>>>;

template <typename T, typename U>
struct IsForwardingAssignmentAmbiguous
    : public absl::conditional_t<
          std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                       U>::value,
          std::false_type,
          IsForwardingAssignmentAmbiguous<
              T, absl::remove_cv_t<absl::remove_reference_t<U>>>> {};

template <typename T, typename U>
struct IsForwardingAssignmentAmbiguous<T, absl::StatusOr<U>>
    : public IsConstructibleOrConvertibleOrAssignableFromStatusOr<T, U> {};

template <typename T, typename U>
using IsForwardingAssignmentValid = absl::disjunction<
    // Short circuits if T is basically U.
    std::is_same<T, absl::remove_cv_t<absl::remove_reference_t<U>>>,
    absl::negation<absl::disjunction<
        std::is_same<absl::StatusOr<T>,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::Status,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::in_place_t,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        IsForwardingAssignmentAmbiguous<T, U>>>>;

template <typename T, typename U>
using IsForwardingAssignmentValid = absl::disjunction<
    // Short circuits if T is basically U.
    std::is_same<T, absl::remove_cv_t<absl::remove_reference_t<U>>>,
    absl::negation<absl::disjunction<
        std::is_same<absl::StatusOr<T>,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::Status,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::in_place_t,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        IsForwardingAssignmentAmbiguous<T, U>>>>;

template <typename T> struct OperatorBase {
  const T &value() const &;
  T &value() &;
  const T &&value() const &&;
  T &&value() &&;

  const T &operator*() const &;
  T &operator*() &;
  const T &&operator*() const &&;
  T &&operator*() &&;

  // To test that analyses are okay if there is a use of operator*
  // within this base class.
  const T *operator->() const { return __builtin_addressof(**this); }
  T *operator->() { return __builtin_addressof(**this); }
};

} // namespace internal_statusor

template <typename T>
struct StatusOr : private internal_statusor::OperatorBase<T> {
  explicit StatusOr();

  StatusOr(const StatusOr &) = default;
  StatusOr &operator=(const StatusOr &) = default;

  StatusOr(StatusOr &&) = default;
  StatusOr &operator=(StatusOr &&) = default;

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, const U &>,
              std::is_convertible<const U &, T>,
              absl::negation<
                  internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                      T, U>>>::value,
          int> = 0>
  StatusOr(const StatusOr<U> &);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, const U &>,
              absl::negation<std::is_convertible<const U &, T>>,
              absl::negation<
                  internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                      T, U>>>::value,
          int> = 0>
  explicit StatusOr(const StatusOr<U> &);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, U &&>, std::is_convertible<U &&, T>,
              absl::negation<
                  internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                      T, U>>>::value,
          int> = 0>
  StatusOr(StatusOr<U> &&);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, U &&>,
              absl::negation<std::is_convertible<U &&, T>>,
              absl::negation<
                  internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                      T, U>>>::value,
          int> = 0>
  explicit StatusOr(StatusOr<U> &&);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, const U &>,
              std::is_assignable<T, const U &>,
              absl::negation<
                  internal_statusor::
                      IsConstructibleOrConvertibleOrAssignableFromStatusOr<
                          T, U>>>::value,
          int> = 0>
  StatusOr &operator=(const StatusOr<U> &);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, U &&>, std::is_assignable<T, U &&>,
              absl::negation<
                  internal_statusor::
                      IsConstructibleOrConvertibleOrAssignableFromStatusOr<
                          T, U>>>::value,
          int> = 0>
  StatusOr &operator=(StatusOr<U> &&);

  template <
      typename U = absl::Status,
      absl::enable_if_t<
          absl::conjunction<
              std::is_convertible<U &&, absl::Status>,
              std::is_constructible<absl::Status, U &&>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::StatusOr<T>>>,
              absl::negation<std::is_same<absl::decay_t<U>, T>>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::in_place_t>>,
              absl::negation<internal_statusor::HasConversionOperatorToStatusOr<
                  T, U &&>>>::value,
          int> = 0>
  StatusOr(U &&);

  template <
      typename U = absl::Status,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_convertible<U &&, absl::Status>>,
              std::is_constructible<absl::Status, U &&>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::StatusOr<T>>>,
              absl::negation<std::is_same<absl::decay_t<U>, T>>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::in_place_t>>,
              absl::negation<internal_statusor::HasConversionOperatorToStatusOr<
                  T, U &&>>>::value,
          int> = 0>
  explicit StatusOr(U &&);

  template <
      typename U = absl::Status,
      absl::enable_if_t<
          absl::conjunction<
              std::is_convertible<U &&, absl::Status>,
              std::is_constructible<absl::Status, U &&>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::StatusOr<T>>>,
              absl::negation<std::is_same<absl::decay_t<U>, T>>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::in_place_t>>,
              absl::negation<internal_statusor::HasConversionOperatorToStatusOr<
                  T, U &&>>>::value,
          int> = 0>
  StatusOr &operator=(U &&);

  template <
      typename U = T,
      typename = typename std::enable_if<absl::conjunction<
          std::is_constructible<T, U &&>, std::is_assignable<T &, U &&>,
          absl::disjunction<
              std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>, T>,
              absl::conjunction<
                  absl::negation<std::is_convertible<U &&, absl::Status>>,
                  absl::negation<
                      internal_statusor::HasConversionOperatorToStatusOr<
                          T, U &&>>>>,
          internal_statusor::IsForwardingAssignmentValid<T, U &&>>::value>::
          type>
  StatusOr &operator=(U &&);

  template <typename... Args> explicit StatusOr(absl::in_place_t, Args &&...);

  template <typename U, typename... Args>
  explicit StatusOr(absl::in_place_t, std::initializer_list<U>, Args &&...);

  template <
      typename U = T,
      absl::enable_if_t<
          absl::conjunction<
              internal_statusor::IsDirectInitializationValid<T, U &&>,
              std::is_constructible<T, U &&>, std::is_convertible<U &&, T>,
              absl::disjunction<
                  std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                               T>,
                  absl::conjunction<
                      absl::negation<std::is_convertible<U &&, absl::Status>>,
                      absl::negation<
                          internal_statusor::HasConversionOperatorToStatusOr<
                              T, U &&>>>>>::value,
          int> = 0>
  StatusOr(U &&);

  template <
      typename U = T,
      absl::enable_if_t<
          absl::conjunction<
              internal_statusor::IsDirectInitializationValid<T, U &&>,
              absl::disjunction<
                  std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                               T>,
                  absl::conjunction<
                      absl::negation<std::is_constructible<absl::Status, U &&>>,
                      absl::negation<
                          internal_statusor::HasConversionOperatorToStatusOr<
                              T, U &&>>>>,
              std::is_constructible<T, U &&>,
              absl::negation<std::is_convertible<U &&, T>>>::value,
          int> = 0>
  explicit StatusOr(U &&);

  bool ok() const;

  const Status &status() const & { return status_; }
  Status status() &&;

  using StatusOr::OperatorBase::value;

  const T &ValueOrDie() const &;
  T &ValueOrDie() &;
  const T &&ValueOrDie() const &&;
  T &&ValueOrDie() &&;

  using StatusOr::OperatorBase::operator*;
  using StatusOr::OperatorBase::operator->;

  template <typename U> T value_or(U &&default_value) const &;
  template <typename U> T value_or(U &&default_value) &&;

  template <typename... Args> T &emplace(Args &&...args);

  template <
      typename U, typename... Args,
      absl::enable_if_t<std::is_constructible<T, std::initializer_list<U> &,
                                              Args &&...>::value,
                        int> = 0>
  T &emplace(std::initializer_list<U> ilist, Args &&...args);

private:
  absl::Status status_;
};

template <typename T>
bool operator==(const StatusOr<T> &lhs, const StatusOr<T> &rhs);

template <typename T>
bool operator!=(const StatusOr<T> &lhs, const StatusOr<T> &rhs);

} // namespace absl

#endif // STATUSOR_H_
)cc";

static constexpr char StdVectorHeader[] = R"cc(
#ifndef STD_VECTOR_H
#define STD_VECTOR_H
namespace std {
template <class T> struct allocator {
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T value_type;

  T *allocate(size_t n);
};

template <class Alloc> struct allocator_traits {
  typedef Alloc allocator_type;
  typedef typename allocator_type::value_type value_type;
  typedef typename allocator_type::pointer pointer;
  typedef typename allocator_type::const_pointer const_pointer;
  typedef typename allocator_type::difference_type difference_type;
  typedef typename allocator_type::size_type size_type;
};

template <typename T, class Allocator = allocator<T>> class vector {
public:
  using value_type = T;
  using size_type = typename allocator_traits<Allocator>::size_type;

  // Constructors.
  vector() {}
  vector(size_type, const Allocator & = Allocator()) {}
  vector(initializer_list<T> initializer_list,
         const Allocator & = Allocator()) {}
  vector(const vector &vector) {}
  ~vector();

  // Modifiers.
  void push_back(const T &value);
  void push_back(T &&value);
  template <typename... Args> T &emplace_back(Args &&...args);

  // Iterators
  class InputIterator {
  public:
    InputIterator(const InputIterator &);
    ~InputIterator();
    InputIterator &operator=(const InputIterator &);
    InputIterator &operator++();
    T &operator*() const;
    bool operator!=(const InputIterator &) const;
    bool operator==(const InputIterator &) const;
  };
  typedef InputIterator iterator;
  typedef const InputIterator const_iterator;
  iterator begin() noexcept;
  const_iterator begin() const noexcept;
  const_iterator cbegin() const noexcept;
  iterator end() noexcept;
  const_iterator end() const noexcept;
  const_iterator cend() const noexcept;
  T *data() noexcept;
  const T *data() const noexcept;
  T &operator[](int n);
  const T &operator[](int n) const;
  T &at(int n);
  const T &at(int n) const;
  size_t size() const;
};
} // namespace std
#endif // STD_VECTOR_H
)cc";

static constexpr char StdPairHeader[] = R"cc(
#ifndef STD_PAIR_H
#define STD_PAIR_H
namespace std {
template <class T1, class T2> struct pair {
  T1 first;
  T2 second;

  typedef T1 first_type;
  typedef T2 second_type;

  constexpr pair();

  template <class U1, class U2> pair(pair<U1, U2> &&p);

  template <class U1, class U2> pair(U1 &&x, U2 &&y);
};

template <class T1, class T2> pair<T1, T2> make_pair(T1 &&t1, T2 &&t2);
} // namespace std
#endif // STD_PAIR_H
)cc";

constexpr const char AbslLogHeader[] = R"cc(
#ifndef ABSL_LOG_H
#define ABSL_LOG_H

#include "std_pair.h"

namespace absl {

#define ABSL_PREDICT_FALSE(x) (__builtin_expect(false || (x), false))
#define ABSL_PREDICT_TRUE(x) (__builtin_expect(false || (x), true))

namespace log_internal {
class LogMessage {
public:
  LogMessage();
  LogMessage &stream();
  LogMessage &InternalStream();
  LogMessage &WithVerbosity(int verboselevel);
  template <typename T> LogMessage &operator<<(const T &);
};
class LogMessageFatal : public LogMessage {
public:
  LogMessageFatal();
  ~LogMessageFatal() __attribute__((noreturn));
};
class LogMessageQuietlyFatal : public LogMessage {
public:
  LogMessageQuietlyFatal();
  ~LogMessageQuietlyFatal() __attribute__((noreturn));
};
class Voidify final {
public:
  // This has to be an operator with a precedence lower than << but higher
  // than
  // ?:
  template <typename T> void operator&&(const T &) const && {}
};
} // namespace log_internal
} // namespace absl

#ifndef NULL
#define NULL __null
#endif
extern "C" void abort() {}
#define ABSL_LOG_INTERNAL_LOG_INFO ::absl::log_internal::LogMessage()
#define ABSL_LOG_INTERNAL_LOG_WARNING ::absl::log_internal::LogMessage()
#define ABSL_LOG_INTERNAL_LOG_ERROR ::absl::log_internal::LogMessage()
#define ABSL_LOG_INTERNAL_LOG_FATAL ::absl::log_internal::LogMessageFatal()
#define ABSL_LOG_INTERNAL_LOG_QFATAL                                           \
  ::absl::log_internal::LogMessageQuietlyFatal()
#define LOG(severity) ABSL_LOG_INTERNAL_LOG_##severity.InternalStream()

#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define ABSL_LOG_INTERNAL_STRIP_STRING_LITERAL(lit) lit

#define ABSL_LOG_INTERNAL_STATELESS_CONDITION(condition)                       \
  switch (0)                                                                   \
  case 0:                                                                      \
  default:                                                                     \
    !(condition) ? (void)0 : ::absl::log_internal::Voidify() &&

#define ABSL_LOG_INTERNAL_CONDITION_INFO(type, condition)                      \
  ABSL_LOG_INTERNAL_##type##_CONDITION(condition)

#define ABSL_LOG_INTERNAL_CONDITION_FATAL(type, condition)                     \
  ABSL_LOG_INTERNAL_##type##_CONDITION(condition)

#define ABSL_LOG_INTERNAL_CONDITION_QFATAL(type, condition)                    \
  ABSL_LOG_INTERNAL_##type##_CONDITION(condition)

#define ABSL_CHECK_IMPL(condition, condition_text)                             \
  ABSL_LOG_INTERNAL_CONDITION_FATAL(STATELESS,                                 \
                                    ABSL_PREDICT_FALSE(!(condition)))          \
  ABSL_LOG_INTERNAL_CHECK(condition_text).InternalStream()

#define ABSL_QCHECK_IMPL(condition, condition_text)                            \
  ABSL_LOG_INTERNAL_CONDITION_QFATAL(STATELESS,                                \
                                     ABSL_PREDICT_FALSE(!(condition)))         \
  ABSL_LOG_INTERNAL_QCHECK(condition_text).InternalStream()

#define CHECK(condition) ABSL_CHECK_IMPL((condition), #condition)
#define DCHECK(condition) CHECK(condition)
#define QCHECK(condition) ABSL_QCHECK_IMPL((condition), #condition)

#define ABSL_LOG_INTERNAL_MAX_LOG_VERBOSITY_CHECK(x)

namespace absl {

template <typename T> class StatusOr;
class Status;

namespace status_internal {
std::string *MakeCheckFailString(const absl::Status *status,
                                 const char *prefix);
} // namespace status_internal

namespace log_internal {
template <class T> const T &GetReferenceableValue(const T &t);
char GetReferenceableValue(char t);
unsigned char GetReferenceableValue(unsigned char t);
signed char GetReferenceableValue(signed char t);
short GetReferenceableValue(short t);
unsigned short GetReferenceableValue(unsigned short t);
int GetReferenceableValue(int t);
unsigned int GetReferenceableValue(unsigned int t);
long GetReferenceableValue(long t);
unsigned long GetReferenceableValue(unsigned long t);
long long GetReferenceableValue(long long t);
unsigned long long GetReferenceableValue(unsigned long long t);
const absl::Status *AsStatus(const absl::Status &s);
template <typename T> const absl::Status *AsStatus(const absl::StatusOr<T> &s);
} // namespace log_internal
} // namespace absl
// TODO(tkd): this still doesn't allow operator<<, unlike the real CHECK_
// macros.
#define ABSL_LOG_INTERNAL_CHECK_OP(name, op, val1, val2)                       \
  while (char *_result = ::absl::log_internal::name##Impl(                     \
             ::absl::log_internal::GetReferenceableValue(val1),                \
             ::absl::log_internal::GetReferenceableValue(val2),                \
             #val1 " " #op " " #val2))                                         \
  (void)0
#define ABSL_LOG_INTERNAL_QCHECK_OP(name, op, val1, val2)                      \
  while (char *_result = ::absl::log_internal::name##Impl(                     \
             ::absl::log_internal::GetReferenceableValue(val1),                \
             ::absl::log_internal::GetReferenceableValue(val2),                \
             #val1 " " #op " " #val2))                                         \
  (void)0
namespace absl {
namespace log_internal {
template <class T1, class T2>
char *Check_NEImpl(const T1 &v1, const T2 &v2, const char *names);
template <class T1, class T2>
char *Check_EQImpl(const T1 &v1, const T2 &v2, const char *names);
template <class T1, class T2>
char *Check_LTImpl(const T1 &v1, const T2 &v2, const char *names);

#define CHECK_EQ(a, b) ABSL_LOG_INTERNAL_CHECK_OP(Check_EQ, ==, a, b)
#define CHECK_NE(a, b) ABSL_LOG_INTERNAL_CHECK_OP(Check_NE, !=, a, b)
#define CHECK_LT(a, b) ABSL_LOG_INTERNAL_CHECK_OP(Check_EQ, <, a, b)

#define QCHECK_EQ(a, b) ABSL_LOG_INTERNAL_QCHECK_OP(Check_EQ, ==, a, b)
#define QCHECK_NE(a, b) ABSL_LOG_INTERNAL_QCHECK_OP(Check_NE, !=, a, b)
} // namespace log_internal
} // namespace absl

#define CHECK_NOTNULL(x) CHECK((x) != nullptr)

#define ABSL_LOG_INTERNAL_CHECK(failure_message)                               \
  ::absl::log_internal::LogMessageFatal()
#define ABSL_LOG_INTERNAL_QCHECK(failure_message)                              \
  ::absl::log_internal::LogMessageQuietlyFatal()
#define ABSL_LOG_INTERNAL_CHECK_OK(val)                                        \
  for (::std::pair<const ::absl::Status *, ::std::string *>                    \
           absl_log_internal_check_ok_goo;                                     \
       absl_log_internal_check_ok_goo.first =                                  \
           ::absl::log_internal::AsStatus(val),                                \
       absl_log_internal_check_ok_goo.second =                                 \
           ABSL_PREDICT_TRUE(absl_log_internal_check_ok_goo.first->ok())       \
               ? nullptr                                                       \
               : ::absl::status_internal::MakeCheckFailString(                 \
                     absl_log_internal_check_ok_goo.first,                     \
                     ABSL_LOG_INTERNAL_STRIP_STRING_LITERAL(#val " is OK")),   \
       !ABSL_PREDICT_TRUE(absl_log_internal_check_ok_goo.first->ok());)        \
  ABSL_LOG_INTERNAL_CHECK(*absl_log_internal_check_ok_goo.second)              \
      .InternalStream()
#define ABSL_LOG_INTERNAL_QCHECK_OK(val)                                       \
  for (::std::pair<const ::absl::Status *, ::std::string *>                    \
           absl_log_internal_check_ok_goo;                                     \
       absl_log_internal_check_ok_goo.first =                                  \
           ::absl::log_internal::AsStatus(val),                                \
       absl_log_internal_check_ok_goo.second =                                 \
           ABSL_PREDICT_TRUE(absl_log_internal_check_ok_goo.first->ok())       \
               ? nullptr                                                       \
               : ::absl::status_internal::MakeCheckFailString(                 \
                     absl_log_internal_check_ok_goo.first,                     \
                     ABSL_LOG_INTERNAL_STRIP_STRING_LITERAL(#val " is OK")),   \
       !ABSL_PREDICT_TRUE(absl_log_internal_check_ok_goo.first->ok());)        \
  ABSL_LOG_INTERNAL_QCHECK(*absl_log_internal_check_ok_goo.second)             \
      .InternalStream()

#define CHECK_OK(val) ABSL_LOG_INTERNAL_CHECK_OK(val)
#define DCHECK_OK(val) ABSL_LOG_INTERNAL_CHECK_OK(val)
#define QCHECK_OK(val) ABSL_LOG_INTERNAL_QCHECK_OK(val)

#endif // ABSL_LOG_H
)cc";

constexpr const char TestingDefsHeader[] = R"cc(
#pragma clang system_header

#ifndef TESTING_DEFS_H
#define TESTING_DEFS_H

#include "absl_type_traits.h"
#include "std_initializer_list.h"
#include "std_string.h"
#include "std_type_traits.h"
#include "std_utility.h"

namespace testing {
struct AssertionResult {
  template <typename T>
  explicit AssertionResult(const T &res, bool enable_if = true) {}
  ~AssertionResult();
  operator bool() const;
  template <typename T> AssertionResult &operator<<(const T &value);
  const char *failure_message() const;
};

class TestPartResult {
public:
  enum Type { kSuccess, kNonFatalFailure, kFatalFailure, kSkip };
};

class Test {
public:
  virtual ~Test() = default;

protected:
  virtual void SetUp() {}
};

class Message {
public:
  template <typename T> Message &operator<<(const T &val);
};

namespace internal {
class AssertHelper {
public:
  AssertHelper(TestPartResult::Type type, const char *file, int line,
               const char *message);
  void operator=(const Message &message) const;
};

class EqHelper {
public:
  template <typename T1, typename T2>
  static AssertionResult Compare(const char *lhx, const char *rhx,
                                 const T1 &lhs, const T2 &rhs);
};

#define GTEST_IMPL_CMP_HELPER_(op_name)                                        \
  template <typename T1, typename T2>                                          \
  AssertionResult CmpHelper##op_name(const char *expr1, const char *expr2,     \
                                     const T1 &val1, const T2 &val2);

GTEST_IMPL_CMP_HELPER_(NE)
GTEST_IMPL_CMP_HELPER_(LE)
GTEST_IMPL_CMP_HELPER_(LT)
GTEST_IMPL_CMP_HELPER_(GE)
GTEST_IMPL_CMP_HELPER_(GT)

#undef GTEST_IMPL_CMP_HELPER_

std::string GetBoolAssertionFailureMessage(
    const AssertionResult &assertion_result, const char *expression_text,
    const char *actual_predicate_value, const char *expected_predicate_value);

template <typename M> class PredicateFormatterFromMatcher {
public:
  template <typename T>
  AssertionResult operator()(const char *value_text, const T &x) const;
};

template <typename M>
inline PredicateFormatterFromMatcher<M>
MakePredicateFormatterFromMatcher(M matcher) {
  return PredicateFormatterFromMatcher<M>();
}
} // namespace internal

namespace status {
namespace internal_status {
class IsOkMatcher {};

class StatusIsMatcher {};

class CanonicalStatusIsMatcher {};

template <typename M> class IsOkAndHoldsMatcher {};

} // namespace internal_status

internal_status::IsOkMatcher IsOk();

template <typename StatusCodeMatcher>
internal_status::StatusIsMatcher StatusIs(StatusCodeMatcher &&code_matcher);

template <typename StatusCodeMatcher>
internal_status::CanonicalStatusIsMatcher
CanonicalStatusIs(StatusCodeMatcher &&code_matcher);

template <typename InnerMatcher>
internal_status::IsOkAndHoldsMatcher<InnerMatcher> IsOkAndHolds(InnerMatcher m);
} // namespace status

class IsTrueMatcher {};
IsTrueMatcher IsTrue();

class IsFalseMatcher {};
IsFalseMatcher IsFalse();

} // namespace testing

namespace absl_testing {
namespace status_internal {
class IsOkMatcher {};
template <typename M> class IsOkAndHoldsMatcher {};
class StatusIsMatcher {};
class CanonicalStatusIsMatcher {};
} // namespace status_internal
status_internal::IsOkMatcher IsOk();
template <typename InnerMatcher>
status_internal::IsOkAndHoldsMatcher<InnerMatcher> IsOkAndHolds(InnerMatcher m);
template <typename StatusCodeMatcher>
status_internal::StatusIsMatcher StatusIs(StatusCodeMatcher &&code_matcher);

template <typename StatusCodeMatcher>
status_internal::CanonicalStatusIsMatcher
CanonicalStatusIs(StatusCodeMatcher &&code_matcher);
} // namespace absl_testing

using testing::AssertionResult;
#define EXPECT_TRUE(x)                                                         \
  switch (0)                                                                   \
  case 0:                                                                      \
  default:                                                                     \
    if (const AssertionResult gtest_ar_ = AssertionResult(x)) {                \
    } else /* NOLINT */                                                        \
      ::testing::Message()
#define EXPECT_FALSE(x) EXPECT_TRUE(!(x))

#define GTEST_AMBIGUOUS_ELSE_BLOCKER_                                          \
  switch (0)                                                                   \
  case 0:                                                                      \
  default:

#define GTEST_ASSERT_(expression, on_failure)                                  \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_                                                \
  if (const ::testing::AssertionResult gtest_ar = (expression))                \
    ;                                                                          \
  else                                                                         \
    on_failure(gtest_ar.failure_message())
#define GTEST_PRED_FORMAT1_(pred_format, v1, on_failure)                       \
  GTEST_ASSERT_(pred_format(#v1, v1), on_failure)
#define GTEST_PRED_FORMAT2_(pred_format, v1, v2, on_failure)                   \
  GTEST_ASSERT_(pred_format(#v1, #v2, v1, v2), on_failure)
#define GTEST_MESSAGE_AT_(file, line, message, result_type)                    \
  ::testing::internal::AssertHelper(result_type, file, line, message) =        \
      ::testing::Message()
#define GTEST_MESSAGE_(message, result_type)                                   \
  GTEST_MESSAGE_AT_(__FILE__, __LINE__, message, result_type)
#define GTEST_FATAL_FAILURE_(message)                                          \
  return GTEST_MESSAGE_(message, ::testing::TestPartResult::kFatalFailure)
#define GTEST_NONFATAL_FAILURE_(message)                                       \
  GTEST_MESSAGE_(message, ::testing::TestPartResult::kNonFatalFailure)

#define ASSERT_PRED_FORMAT1(pred_format, v1)                                   \
  GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED_FORMAT2(pred_format, v1, v2)                               \
  GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_FATAL_FAILURE_)

#define ASSERT_THAT(value, matcher)                                            \
  ASSERT_PRED_FORMAT1(                                                         \
      ::testing::internal::MakePredicateFormatterFromMatcher(matcher), value)
#define ASSERT_OK(x) ASSERT_THAT(x, ::testing::status::IsOk())

#define EXPECT_PRED_FORMAT1(pred_format, v1)                                   \
  GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED_FORMAT2(pred_format, v1, v2)                               \
  GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_NONFATAL_FAILURE_)
#define EXPECT_THAT(value, matcher)                                            \
  EXPECT_PRED_FORMAT1(                                                         \
      ::testing::internal::MakePredicateFormatterFromMatcher(matcher), value)
#define EXPECT_OK(expression) EXPECT_THAT(expression, ::testing::status::IsOk())

#define GTEST_TEST_BOOLEAN_(expression, text, actual, expected, fail)          \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_                                                \
  if (const ::testing::AssertionResult gtest_ar_ =                             \
          ::testing::AssertionResult(expression))                              \
    ;                                                                          \
  else                                                                         \
    fail(::testing::internal::GetBoolAssertionFailureMessage(                  \
             gtest_ar_, text, #actual, #expected)                              \
             .c_str())
#define GTEST_ASSERT_TRUE(condition)                                           \
  GTEST_TEST_BOOLEAN_(condition, #condition, false, true, GTEST_FATAL_FAILURE_)
#define GTEST_ASSERT_FALSE(condition)                                          \
  GTEST_TEST_BOOLEAN_(!(condition), #condition, true, false,                   \
                      GTEST_FATAL_FAILURE_)
#define ASSERT_TRUE(condition) GTEST_ASSERT_TRUE(condition)
#define ASSERT_FALSE(condition) GTEST_ASSERT_FALSE(condition)

#define EXPECT_EQ(x, y)                                                        \
  EXPECT_PRED_FORMAT2(::testing::internal::EqHelper::Compare, x, y)
#define EXPECT_NE(x, y)                                                        \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperNE, x, y)
#define EXPECT_LT(x, y)                                                        \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLT, x, y)
#define EXPECT_GT(x, y)                                                        \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGT, x, y)
#define EXPECT_LE(x, y)                                                        \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLE, x, y)
#define EXPECT_GE(x, y)                                                        \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGE, x, y)

#define ASSERT_EQ(x, y)                                                        \
  ASSERT_PRED_FORMAT2(testing::internal::EqHelper::Compare, x, y)
#define ASSERT_NE(x, y)                                                        \
  ASSERT_PRED_FORMAT2(testing::internal::CmpHelperNE, x, y)
#define ASSERT_LT(x, y)                                                        \
  ASSERT_PRED_FORMAT2(testing::internal::CmpHelperLT, x, y)
#define ASSERT_GT(x, y)                                                        \
  ASSERT_PRED_FORMAT2(testing::internal::CmpHelperGT, x, y)
#define ASSERT_LE(x, y)                                                        \
  ASSERT_PRED_FORMAT2(testing::internal::CmpHelperLE, x, y)
#define ASSERT_GE(x, y)                                                        \
  ASSERT_PRED_FORMAT2(testing::internal::CmpHelperGE, x, y)

#endif // TESTING_DEFS_H
)cc";

std::vector<std::pair<std::string, std::string>> getMockHeaders() {
  std::vector<std::pair<std::string, std::string>> Headers;
  Headers.emplace_back("cstddef.h", CStdDefHeader);
  Headers.emplace_back("std_initializer_list.h", StdInitializerListHeader);
  Headers.emplace_back("std_string.h", StdStringHeader);
  Headers.emplace_back("std_type_traits.h", StdTypeTraitsHeader);
  Headers.emplace_back("std_utility.h", StdUtilityHeader);
  Headers.emplace_back("std_optional.h", StdOptionalHeader);
  Headers.emplace_back("absl_type_traits.h", AbslTypeTraitsHeader);
  Headers.emplace_back("absl_optional.h", AbslOptionalHeader);
  Headers.emplace_back("base_optional.h", BaseOptionalHeader);
  Headers.emplace_back("std_vector.h", StdVectorHeader);
  Headers.emplace_back("std_pair.h", StdPairHeader);
  Headers.emplace_back("status_defs.h", StatusDefsHeader);
  Headers.emplace_back("statusor_defs.h", StatusOrDefsHeader);
  Headers.emplace_back("absl_log.h", AbslLogHeader);
  Headers.emplace_back("testing_defs.h", TestingDefsHeader);
  return Headers;
}

} // namespace test
} // namespace dataflow
} // namespace clang