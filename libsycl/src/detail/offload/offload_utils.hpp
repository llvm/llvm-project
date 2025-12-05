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

const char *stringifyErrorCode(ol_errc_t error);

inline std::string formatCodeString(ol_result_t Result) {
  return std::to_string(Result->Code) + " (" +
         std::string(stringifyErrorCode(Result->Code)) + ")" + Result->Details;
}

template <sycl::errc errc = sycl::errc::runtime>
void checkAndThrow(ol_result_t Result) {
  if (Result != OL_SUCCESS) {
    throw sycl::exception(sycl::make_error_code(errc),
                          detail::formatCodeString(Result));
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

/// Helper to map SYCL information descriptors to OL_<HANDLE>_INFO_<SMTH>. To be
/// used like:
///
/// using Map = info_ol_mapping<ol_foo_info_t>;
/// constexpr auto olInfo = map_info_desc<FromDesc, ol_foo_info_t>(
///                                            Map::M<DescVal0>{OL_FOO_INFO_VAL0},
///                                            Map::M<DescVal1>{OL_FOO_INFO_VAL1},
///                                            ...)
template <typename To> struct info_ol_mapping {
  template <typename From> struct M {
    To value;
    constexpr M(To value) : value(value) {}
  };
};
template <typename From, typename To, typename... Ts>
constexpr To map_info_desc(typename info_ol_mapping<To>::template M<Ts>... ms) {
  return std::get<typename info_ol_mapping<To>::template M<From>>(
             std::tuple{ms...})
      .value;
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_OFFLOAD_UTILS
