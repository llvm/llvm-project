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

/// Converts liboffload error code to C-string.
///
/// \param Error liboffload error code.
///
/// \returns C-string representing the name of Error as specified in enum.
const char *stringifyErrorCode(ol_errc_t Error);

/// Contructs C++-string with information about liboffload error.
///
/// \param Error liboffload result of calling API.
///
/// \returns C++-string containing all available data of failure.
inline std::string formatCodeString(ol_result_t Result) {
  return std::to_string(Result->Code) + " (" +
         std::string(stringifyErrorCode(Result->Code)) + ") " + Result->Details;
}

/// Checks liboffload API call result.
///
/// Used after calling the API without check.
/// To be called when specific handling is needed and explicitly done by
/// developer before throwing exception.
///
/// \param Error liboffload result of calling API.
///
/// \throw sycl::runtime_exception if the call was not successful.
template <sycl::errc errc = sycl::errc::runtime>
void checkAndThrow(ol_result_t Result) {
  if (Result != OL_SUCCESS) {
    throw sycl::exception(sycl::make_error_code(errc),
                          detail::formatCodeString(Result));
  }
}

/// Calls the API, doesn't check result.
/// To be called when specific handling is needed and explicitly done by
/// developer after.
///
/// \param Function liboffload API function to be called.
/// \param Args arguments to be passed to the liboffload API function.
///
/// \returns liboffload error code returned by API call.
template <typename FunctionType, typename... ArgsT>
ol_result_t callNoCheck(FunctionType &Function, ArgsT &&...Args) {
  return Function(std::forward<ArgsT>(Args)...);
}

/// Calls the API and checks result.
///
/// \param Function liboffload API function to be called.
/// \param Args arguments to be passed to the liboffload API function.
///
/// \throw sycl::runtime_exception if the call was not successful.
template <typename FunctionType, typename... ArgsT>
void callAndThrow(FunctionType &Function, ArgsT &&...Args) {
  auto Err = callNoCheck(Function, std::forward<ArgsT>(Args)...);
  checkAndThrow(Err);
}

/// Converts liboffload backend to SYCL backend.
///
/// \param Backend liboffload backend.
///
/// \returns sycl::backend matching specified liboffload backend.
backend convertBackend(ol_platform_backend_t Backend);

/// Helper to map SYCL information descriptors to OL_<HANDLE>_INFO_<SMTH>.
///
/// Typical usage:
/// \code
///   using Map = info_ol_mapping<ol_foo_info_t>;
///   constexpr auto olInfo = map_info_desc<FromDesc, ol_foo_info_t>(
///                                            Map::M<DescVal0>{OL_FOO_INFO_VAL0},
///                                            Map::M<DescVal1>{OL_FOO_INFO_VAL1},
///                                          ...)
/// \endcode
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
