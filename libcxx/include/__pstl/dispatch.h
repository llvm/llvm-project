//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_DISPATCH_H
#define _LIBCPP___PSTL_DISPATCH_H

#include <__config>
#include <__pstl/backend_fwd.h>
#include <__type_traits/conditional.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_callable.h>
#include <__type_traits/type_identity.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <template <class, class> class _Algorithm, class _Backend, class _ExecutionPolicy, class = void>
constexpr bool __is_implemented_v = false;

template <template <class, class> class _Algorithm, class _Backend, class _ExecutionPolicy>
constexpr bool __is_implemented_v<_Algorithm,
                                  _Backend,
                                  _ExecutionPolicy,
                                  __enable_if_t<sizeof(_Algorithm<_Backend, _ExecutionPolicy>)>> = true;

// Helpful to provide better error messages. This will show the algorithm and the execution policy
// in the compiler diagnostic.
template <template <class, class> class _Algorithm, class _ExecutionPolicy>
constexpr bool __cant_find_backend_for = false;

template <template <class, class> class _Algorithm, class _BackendConfiguration, class _ExecutionPolicy>
struct __find_first_implemented;

template <template <class, class> class _Algorithm, class _ExecutionPolicy>
struct __find_first_implemented<_Algorithm, __backend_configuration<>, _ExecutionPolicy> {
  static_assert(__cant_find_backend_for<_Algorithm, _ExecutionPolicy>,
                "Could not find a PSTL backend for the given algorithm and execution policy");
};

template <class _Backend, class _RemainingBackends>
struct __backend_info {
  using __backend            = _Backend;
  using __remaining_backends = _RemainingBackends;
};

template <template <class, class> class _Algorithm, class _B1, class... _Bn, class _ExecutionPolicy>
struct __find_first_implemented<_Algorithm, __backend_configuration<_B1, _Bn...>, _ExecutionPolicy>
    : _If<__is_implemented_v<_Algorithm, _B1, _ExecutionPolicy>,
          __backend_info<_Algorithm<_B1, _ExecutionPolicy>, __backend_configuration<_Bn...>>,
          __find_first_implemented<_Algorithm, __backend_configuration<_Bn...>, _ExecutionPolicy> > {};

template <template <class, class> class _Algorithm, class _BackendConfiguration, class _ExecutionPolicy>
struct __dispatch {
  template <class... _Args>
  [[nodiscard]] auto operator()(_Args&&... __args) {
    using _NextBackend = __find_first_implemented<_Algorithm, _BackendConfiguration, _ExecutionPolicy>;
    if constexpr (__is_callable<typename _NextBackend::__backend, _Args...>::value) {
      return typename _NextBackend::__backend()(std::forward<_Args>(__args)...);
    } else {
      return __dispatch<_Algorithm, typename _NextBackend::__remaining_backends, _ExecutionPolicy>()(
          std::forward<_Args>(__args)...);
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_DISPATCH_H
