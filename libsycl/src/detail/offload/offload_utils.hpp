//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_OFFLOAD_UTILS
#define _LIBSYCL_OFFLOAD_UTILS

#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/exception.hpp>

#include <OffloadAPI.h>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

const char *stringifyErrorCode(int32_t error);

inline std::string formatCodeString(int32_t code) {
  return std::to_string(code) + " (" + std::string(stringifyErrorCode(code)) +
         ")";
}

template <sycl::errc errc = sycl::errc::runtime>
void checkAndThrow(ol_result_t Result) {
  if (Result != OL_SUCCESS) {
    throw sycl::exception(sycl::make_error_code(errc),
                          detail::formatCodeString(Result->Code));
  }
}

/// Calls the API, doesn't check result. To be called when specific handling is
/// needed and explicitly done by developer after.
template <typename FunctionType, typename... ArgsT>
ol_result_t call_nocheck(FunctionType &Function, ArgsT &&...Args) {
  return Function(std::forward<ArgsT>(Args)...);
}

/// Calls the API & checks the result
///
/// \throw sycl::runtime_exception if the call was not successful.
template <typename FunctionType, typename... ArgsT>
void call_and_throw(FunctionType &Function, ArgsT &&...Args) {
  auto Err = call_nocheck(Function, std::forward<ArgsT>(Args)...);
  checkAndThrow(Err);
}

backend convertBackend(ol_platform_backend_t Backend);

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_OFFLOAD_UTILS
