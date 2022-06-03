//===- UncheckedOptionalAccessModelTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// FIXME: Move this to clang/unittests/Analysis/FlowSensitive/Models.

#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/SourceLocationsLattice.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace dataflow;
using namespace test;

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

// FIXME: Move header definitions in separate file(s).
static constexpr char CSDtdDefHeader[] = R"(
#ifndef CSTDDEF_H
#define CSTDDEF_H

namespace std {

typedef decltype(sizeof(char)) size_t;

using nullptr_t = decltype(nullptr);

} // namespace std

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
auto try_add_rvalue_reference(int) -> type_identity<T&&>;
template <class T>
auto try_add_rvalue_reference(...) -> type_identity<T>;

}  // namespace detail

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

template <typename T>
struct negation : std::integral_constant<bool, !T::value> {};

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

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

struct in_place_t {};
constexpr in_place_t in_place;

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

} // namespace base
)";

/// Converts `L` to string.
static std::string ConvertToString(const SourceLocationsLattice &L,
                                   const ASTContext &Ctx) {
  return L.getSourceLocations().empty() ? "safe"
                                        : "unsafe: " + DebugString(L, Ctx);
}

/// Replaces all occurrences of `Pattern` in `S` with `Replacement`.
static void ReplaceAllOccurrences(std::string &S, const std::string &Pattern,
                                  const std::string &Replacement) {
  size_t Pos = 0;
  while (true) {
    Pos = S.find(Pattern, Pos);
    if (Pos == std::string::npos)
      break;
    S.replace(Pos, Pattern.size(), Replacement);
  }
}

struct OptionalTypeIdentifier {
  std::string NamespaceName;
  std::string TypeName;
};

class UncheckedOptionalAccessTest
    : public ::testing::TestWithParam<OptionalTypeIdentifier> {
protected:
  template <typename LatticeChecksMatcher>
  void ExpectLatticeChecksFor(std::string SourceCode,
                              LatticeChecksMatcher MatchesLatticeChecks) {
    ExpectLatticeChecksFor(SourceCode, ast_matchers::hasName("target"),
                           MatchesLatticeChecks);
  }

private:
  template <typename FuncDeclMatcher, typename LatticeChecksMatcher>
  void ExpectLatticeChecksFor(std::string SourceCode,
                              FuncDeclMatcher FuncMatcher,
                              LatticeChecksMatcher MatchesLatticeChecks) {
    ReplaceAllOccurrences(SourceCode, "$ns", GetParam().NamespaceName);
    ReplaceAllOccurrences(SourceCode, "$optional", GetParam().TypeName);

    std::vector<std::pair<std::string, std::string>> Headers;
    Headers.emplace_back("cstddef.h", CSDtdDefHeader);
    Headers.emplace_back("std_initializer_list.h", StdInitializerListHeader);
    Headers.emplace_back("std_string.h", StdStringHeader);
    Headers.emplace_back("std_type_traits.h", StdTypeTraitsHeader);
    Headers.emplace_back("std_utility.h", StdUtilityHeader);
    Headers.emplace_back("std_optional.h", StdOptionalHeader);
    Headers.emplace_back("absl_type_traits.h", AbslTypeTraitsHeader);
    Headers.emplace_back("absl_optional.h", AbslOptionalHeader);
    Headers.emplace_back("base_optional.h", BaseOptionalHeader);
    Headers.emplace_back("unchecked_optional_access_test.h", R"(
      #include "absl_optional.h"
      #include "base_optional.h"
      #include "std_initializer_list.h"
      #include "std_optional.h"
      #include "std_string.h"
      #include "std_utility.h"

      template <typename T>
      T Make();
    )");
    const tooling::FileContentMappings FileContents(Headers.begin(),
                                                    Headers.end());
    llvm::Error Error = checkDataflow<UncheckedOptionalAccessModel>(
        SourceCode, FuncMatcher,
        [](ASTContext &Ctx, Environment &) {
          return UncheckedOptionalAccessModel(
              Ctx, UncheckedOptionalAccessModelOptions{
                       /*IgnoreSmartPointerDereference=*/true});
        },
        [&MatchesLatticeChecks](
            llvm::ArrayRef<std::pair<
                std::string, DataflowAnalysisState<SourceLocationsLattice>>>
                CheckToLatticeMap,
            ASTContext &Ctx) {
          // FIXME: Consider using a matcher instead of translating
          // `CheckToLatticeMap` to `CheckToStringifiedLatticeMap`.
          std::vector<std::pair<std::string, std::string>>
              CheckToStringifiedLatticeMap;
          for (const auto &E : CheckToLatticeMap) {
            CheckToStringifiedLatticeMap.emplace_back(
                E.first, ConvertToString(E.second.Lattice, Ctx));
          }
          EXPECT_THAT(CheckToStringifiedLatticeMap, MatchesLatticeChecks);
        },
        {"-fsyntax-only", "-std=c++17", "-Wno-undefined-inline"}, FileContents);
    if (Error)
      FAIL() << llvm::toString(std::move(Error));
  }
};

INSTANTIATE_TEST_SUITE_P(
    UncheckedOptionalUseTestInst, UncheckedOptionalAccessTest,
    ::testing::Values(OptionalTypeIdentifier{"std", "optional"},
                      OptionalTypeIdentifier{"absl", "optional"},
                      OptionalTypeIdentifier{"base", "Optional"}),
    [](const ::testing::TestParamInfo<OptionalTypeIdentifier> &Info) {
      return Info.param.NamespaceName;
    });

TEST_P(UncheckedOptionalAccessTest, EmptyFunctionBody) {
  ExpectLatticeChecksFor(R"(
    void target() {
      (void)0;
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, UnwrapUsingValueNoCheck) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      std::move(opt).value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:7")));
}

TEST_P(UncheckedOptionalAccessTest, UnwrapUsingOperatorStarNoCheck) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      *opt;
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:8")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      *std::move(opt);
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:8")));
}

TEST_P(UncheckedOptionalAccessTest, UnwrapUsingOperatorArrowNoCheck) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      void foo();
    };

    void target($ns::$optional<Foo> opt) {
      opt->foo();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:9:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      void foo();
    };

    void target($ns::$optional<Foo> opt) {
      std::move(opt)->foo();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:9:7")));
}

TEST_P(UncheckedOptionalAccessTest, HasValueCheck) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      if (opt.has_value()) {
        opt.value();
        /*[[check]]*/
      }
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, OperatorBoolCheck) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      if (opt) {
        opt.value();
        /*[[check]]*/
      }
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, UnwrapFunctionCallResultNoCheck) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      Make<$ns::$optional<int>>().value();
      (void)0;
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      std::move(opt).value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:7")));
}

TEST_P(UncheckedOptionalAccessTest, DefaultConstructor) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt;
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:6:7")));
}

TEST_P(UncheckedOptionalAccessTest, NulloptConstructor) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt($ns::nullopt);
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:6:7")));
}

TEST_P(UncheckedOptionalAccessTest, InPlaceConstructor) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt($ns::in_place, 3);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    void target() {
      $ns::$optional<Foo> opt($ns::in_place);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      explicit Foo(int, bool);
    };

    void target() {
      $ns::$optional<Foo> opt($ns::in_place, 3, false);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      explicit Foo(std::initializer_list<int>);
    };

    void target() {
      $ns::$optional<Foo> opt($ns::in_place, {3});
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, ValueConstructor) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt(21);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::$optional<int>(21);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<$ns::$optional<int>> opt(Make<$ns::$optional<int>>());
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct MyString {
      MyString(const char*);
    };

    void target() {
      $ns::$optional<MyString> opt("foo");
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Bar> opt(Make<Foo>());
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      explicit Foo(int);
    };

    void target() {
      $ns::$optional<Foo> opt(3);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, ConvertibleOptionalConstructor) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Bar> opt(Make<$ns::$optional<Foo>>());
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:12:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      explicit Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Bar> opt(Make<$ns::$optional<Foo>>());
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:12:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1 = $ns::nullopt;
      $ns::$optional<Bar> opt2(opt1);
      opt2.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:13:7")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1(Make<Foo>());
      $ns::$optional<Bar> opt2(opt1);
      opt2.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      explicit Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1(Make<Foo>());
      $ns::$optional<Bar> opt2(opt1);
      opt2.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, MakeOptional) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::make_optional(0);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      Foo(int, int);
    };

    void target() {
      $ns::$optional<Foo> opt = $ns::make_optional<Foo>(21, 22);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      constexpr Foo(std::initializer_list<char>);
    };

    void target() {
      char a = 'a';
      $ns::$optional<Foo> opt = $ns::make_optional<Foo>({a});
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, ValueOr) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt;
      opt.value_or(0);
      (void)0;
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, ValueOrComparison) {
  // Pointers.
  ExpectLatticeChecksFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int*> opt) {
      if (opt.value_or(nullptr) != nullptr) {
        opt.value();
        /*[[check-ptrs-1]]*/
      } else {
        opt.value();
        /*[[check-ptrs-2]]*/
      }
    }
  )code",
      UnorderedElementsAre(Pair("check-ptrs-1", "safe"),
                           Pair("check-ptrs-2", "unsafe: input.cc:9:9")));

  // Integers.
  ExpectLatticeChecksFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      if (opt.value_or(0) != 0) {
        opt.value();
        /*[[check-ints-1]]*/
      } else {
        opt.value();
        /*[[check-ints-2]]*/
      }
    }
  )code",
      UnorderedElementsAre(Pair("check-ints-1", "safe"),
                           Pair("check-ints-2", "unsafe: input.cc:9:9")));

  // Strings.
  ExpectLatticeChecksFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<std::string> opt) {
      if (!opt.value_or("").empty()) {
        opt.value();
        /*[[check-strings-1]]*/
      } else {
        opt.value();
        /*[[check-strings-2]]*/
      }
    }
  )code",
      UnorderedElementsAre(Pair("check-strings-1", "safe"),
                           Pair("check-strings-2", "unsafe: input.cc:9:9")));

  ExpectLatticeChecksFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<std::string> opt) {
      if (opt.value_or("") != "") {
        opt.value();
        /*[[check-strings-neq-1]]*/
      } else {
        opt.value();
        /*[[check-strings-neq-2]]*/
      }
    }
  )code",
      UnorderedElementsAre(
          Pair("check-strings-neq-1", "safe"),
          Pair("check-strings-neq-2", "unsafe: input.cc:9:9")));

  // Pointer-to-optional.
  //
  // FIXME: make `opt` a parameter directly, once we ensure that all `optional`
  // values have a `has_value` property.
  ExpectLatticeChecksFor(
      R"code(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> p) {
      $ns::$optional<int> *opt = &p;
      if (opt->value_or(0) != 0) {
        opt->value();
        /*[[check-pto-1]]*/
      } else {
        opt->value();
        /*[[check-pto-2]]*/
      }
    }
  )code",
      UnorderedElementsAre(Pair("check-pto-1", "safe"),
                           Pair("check-pto-2", "unsafe: input.cc:10:9")));
}

TEST_P(UncheckedOptionalAccessTest, Emplace) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt;
      opt.emplace(0);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> *opt) {
      opt->emplace(0);
      opt->value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  // FIXME: Add tests that call `emplace` in conditional branches.
}

TEST_P(UncheckedOptionalAccessTest, Reset) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::make_optional(0);
      opt.reset();
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:7:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> &opt) {
      if (opt.has_value()) {
        opt.reset();
        opt.value();
        /*[[check]]*/
      }
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:7:9")));

  // FIXME: Add tests that call `reset` in conditional branches.
}

TEST_P(UncheckedOptionalAccessTest, ValueAssignment) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    void target() {
      $ns::$optional<Foo> opt;
      opt = Foo();
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    void target() {
      $ns::$optional<Foo> opt;
      (opt = Foo()).value();
      (void)0;
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct MyString {
      MyString(const char*);
    };

    void target() {
      $ns::$optional<MyString> opt;
      opt = "foo";
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct MyString {
      MyString(const char*);
    };

    void target() {
      $ns::$optional<MyString> opt;
      (opt = "foo").value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, OptionalConversionAssignment) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1 = Foo();
      $ns::$optional<Bar> opt2;
      opt2 = opt1;
      opt2.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1;
      $ns::$optional<Bar> opt2;
      if (opt2.has_value()) {
        opt2 = opt1;
        opt2.value();
        /*[[check]]*/
      }
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:15:9")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {};

    struct Bar {
      Bar(const Foo&);
    };

    void target() {
      $ns::$optional<Foo> opt1 = Foo();
      $ns::$optional<Bar> opt2;
      (opt2 = opt1).value();
      (void)0;
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, NulloptAssignment) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = 3;
      opt = $ns::nullopt;
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:7:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = 3;
      (opt = $ns::nullopt).value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:6:7")));
}

TEST_P(UncheckedOptionalAccessTest, OptionalSwap) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = 3;

      opt1.swap(opt2);

      opt1.value();
      /*[[check-1]]*/

      opt2.value();
      /*[[check-2]]*/
    }
  )",
      UnorderedElementsAre(Pair("check-1", "safe"),
                           Pair("check-2", "unsafe: input.cc:13:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = 3;

      opt2.swap(opt1);

      opt1.value();
      /*[[check-3]]*/

      opt2.value();
      /*[[check-4]]*/
    }
  )",
      UnorderedElementsAre(Pair("check-3", "safe"),
                           Pair("check-4", "unsafe: input.cc:13:7")));
}

TEST_P(UncheckedOptionalAccessTest, StdSwap) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = 3;

      std::swap(opt1, opt2);

      opt1.value();
      /*[[check-1]]*/

      opt2.value();
      /*[[check-2]]*/
    }
  )",
      UnorderedElementsAre(Pair("check-1", "safe"),
                           Pair("check-2", "unsafe: input.cc:13:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt1 = $ns::nullopt;
      $ns::$optional<int> opt2 = 3;

      std::swap(opt2, opt1);

      opt1.value();
      /*[[check-3]]*/

      opt2.value();
      /*[[check-4]]*/
    }
  )",
      UnorderedElementsAre(Pair("check-3", "safe"),
                           Pair("check-4", "unsafe: input.cc:13:7")));
}

TEST_P(UncheckedOptionalAccessTest, UniquePtrToStructWithOptionalField) {
  // We suppress diagnostics for values reachable from smart pointers (other
  // than `optional` itself).
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    template <typename T>
    struct smart_ptr {
      T& operator*() &;
      T* operator->();
    };

    struct Foo {
      $ns::$optional<int> opt;
    };

    void target() {
      smart_ptr<Foo> foo;
      *foo->opt;
      /*[[check-1]]*/
      *(*foo).opt;
      /*[[check-2]]*/
    }
  )",
      UnorderedElementsAre(Pair("check-1", "safe"), Pair("check-2", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, CallReturningOptional) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    $ns::$optional<int> MakeOpt();

    void target() {
      $ns::$optional<int> opt = 0;
      opt = MakeOpt();
      opt.value();
      /*[[check-1]]*/
    }
  )",
      UnorderedElementsAre(Pair("check-1", "unsafe: input.cc:9:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    const $ns::$optional<int>& MakeOpt();

    void target() {
      $ns::$optional<int> opt = 0;
      opt = MakeOpt();
      opt.value();
      /*[[check-2]]*/
    }
  )",
      UnorderedElementsAre(Pair("check-2", "unsafe: input.cc:9:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    using IntOpt = $ns::$optional<int>;
    IntOpt MakeOpt();

    void target() {
      IntOpt opt = 0;
      opt = MakeOpt();
      opt.value();
      /*[[check-3]]*/
    }
  )",
      UnorderedElementsAre(Pair("check-3", "unsafe: input.cc:10:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    using IntOpt = $ns::$optional<int>;
    const IntOpt& MakeOpt();

    void target() {
      IntOpt opt = 0;
      opt = MakeOpt();
      opt.value();
      /*[[check-4]]*/
    }
  )",
      UnorderedElementsAre(Pair("check-4", "unsafe: input.cc:10:7")));
}

// FIXME: Add support for:
// - constructors (copy, move)
// - assignment operators (default, copy, move)
// - invalidation (passing optional by non-const reference/pointer)
// - nested `optional` values
