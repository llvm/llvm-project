// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_TEST_EXPERIMENTAL_TASK_AWAITABLE_TRAITS
#define _LIBCPP_TEST_EXPERIMENTAL_TASK_AWAITABLE_TRAITS

#include <type_traits>
#include <experimental/coroutine>

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_COROUTINES

template<typename _Tp>
struct __is_coroutine_handle : std::false_type {};

template<typename _Tp>
struct __is_coroutine_handle<std::experimental::coroutine_handle<_Tp>> :
  std::true_type
{};

template<typename _Tp>
struct __is_valid_await_suspend_result :
  std::disjunction<
    std::is_void<_Tp>,
    std::is_same<_Tp, bool>,
    __is_coroutine_handle<_Tp>>
{};

template<typename _Tp, typename = void>
struct is_awaiter : std::false_type {};

template<typename _Tp>
struct is_awaiter<_Tp, std::void_t<
  decltype(std::declval<_Tp&>().await_ready()),
  decltype(std::declval<_Tp&>().await_resume()),
  decltype(std::declval<_Tp&>().await_suspend(
    std::declval<std::experimental::coroutine_handle<void>>()))>> :
  std::conjunction<
    std::is_same<decltype(std::declval<_Tp&>().await_ready()), bool>,
    __is_valid_await_suspend_result<decltype(
      std::declval<_Tp&>().await_suspend(
        std::declval<std::experimental::coroutine_handle<void>>()))>>
{};

template<typename _Tp>
constexpr bool is_awaiter_v = is_awaiter<_Tp>::value;

template<typename _Tp, typename = void>
struct __has_member_operator_co_await : std::false_type {};

template<typename _Tp>
struct __has_member_operator_co_await<_Tp, std::void_t<decltype(std::declval<_Tp>().operator co_await())>>
: is_awaiter<decltype(std::declval<_Tp>().operator co_await())>
{};

template<typename _Tp, typename = void>
struct __has_non_member_operator_co_await : std::false_type {};

template<typename _Tp>
struct __has_non_member_operator_co_await<_Tp, std::void_t<decltype(operator co_await(std::declval<_Tp>()))>>
: is_awaiter<decltype(operator co_await(std::declval<_Tp>()))>
{};

template<typename _Tp>
struct is_awaitable : std::disjunction<
  is_awaiter<_Tp>,
  __has_member_operator_co_await<_Tp>,
  __has_non_member_operator_co_await<_Tp>>
{};

template<typename _Tp>
constexpr bool is_awaitable_v = is_awaitable<_Tp>::value;

template<
  typename _Tp,
  std::enable_if_t<is_awaitable_v<_Tp>, int> = 0>
decltype(auto) get_awaiter(_Tp&& __awaitable)
{
  if constexpr (__has_member_operator_co_await<_Tp>::value)
  {
    return static_cast<_Tp&&>(__awaitable).operator co_await();
  }
  else if constexpr (__has_non_member_operator_co_await<_Tp>::value)
  {
    return operator co_await(static_cast<_Tp&&>(__awaitable));
  }
  else
  {
    return static_cast<_Tp&&>(__awaitable);
  }
}

template<typename _Tp, typename = void>
struct await_result
{};

template<typename _Tp>
struct await_result<_Tp, std::enable_if_t<is_awaitable_v<_Tp>>>
{
private:
  using __awaiter = decltype(get_awaiter(std::declval<_Tp>()));
public:
  using type = decltype(std::declval<__awaiter&>().await_resume());
};

template<typename _Tp>
using await_result_t = typename await_result<_Tp>::type;

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_COROUTINES

#endif
