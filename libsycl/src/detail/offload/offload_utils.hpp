//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of helpers for libsycl-to-liboffload
/// communication and data conversion.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_SRC_DETAIL_OFFLOAD_OFFLOAD_UTILS_HPP
#define _LIBSYCL_SRC_DETAIL_OFFLOAD_OFFLOAD_UTILS_HPP

#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/context.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/unified_range_view.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/info/device_type.hpp>
#include <sycl/__impl/usm_alloc_type.hpp>

#include <OffloadAPI.h>

#include <string>
#include <tuple>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class ContextImpl;

/// Converts a liboffload error code to its name.
///
/// \param Error is the liboffload error code.
/// \return the name of \p Error as spelled in the ol_errc_t enum.
const char *stringifyErrorCode(ol_errc_t Error);

/// Formats a failed liboffload result into an error message.
///
/// \param Result is the liboffload result of a failed API call.
/// \return a string with all available data about the failure.
inline std::string formatCodeString(ol_result_t Result) {
  // liboffload leaves Details unset for errors it has no message for.
  const char *Details = Result->Details ? Result->Details : "";
  return std::to_string(Result->Code) + " (" +
         std::string(stringifyErrorCode(Result->Code)) + ") " + Details;
}

inline bool isFailed(const ol_result_t &Result) { return Result != OL_SUCCESS; }

/// Checks the result of a liboffload API call that was made without a check,
/// when specific handling is needed and explicitly done by the developer before
/// throwing an exception.
///
/// \param Result the liboffload result of calling the API.
/// \throw sycl::exception with ErrC if the call was not successful.
template <sycl::errc ErrC = sycl::errc::runtime>
void checkAndThrow(ol_result_t Result) {
  if (isFailed(Result)) {
    throw sycl::exception(sycl::make_error_code(ErrC),
                          detail::formatCodeString(Result));
  }
}

/// \overload
///
/// \param Context is the context the API call was made for, which is attached
/// to the exception.
/// \param Result is the liboffload result of calling the API.
/// \throw sycl::exception with ErrC if the call was not successful.
template <sycl::errc ErrC = sycl::errc::runtime>
void checkAndThrow(ContextImpl &Context, ol_result_t Result) {
  if (isFailed(Result)) {
    throw sycl::exception(createSyclObjFromImpl<sycl::context>(Context),
                          sycl::make_error_code(ErrC),
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
/// \return liboffload error code returned by API call.
template <typename FunctionType, typename... ArgsT>
ol_result_t callNoCheck(FunctionType &Function, ArgsT &&...Args) {
  return Function(std::forward<ArgsT>(Args)...);
}

/// Calls the liboffload API and checks the result.
///
/// \param Function the liboffload API function to be called.
/// \param Args the arguments to be passed to the liboffload API function.
/// \throw sycl::exception with sycl::errc::runtime if the call was not
/// successful.
template <typename FunctionType, typename... ArgsT>
void callAndThrow(FunctionType &Function, ArgsT &&...Args) {
  auto Err = callNoCheck(Function, std::forward<ArgsT>(Args)...);
  checkAndThrow(Err);
}

/// \overload
///
/// \param Context the context the API call is made for, which is attached to
/// the exception.
/// \param Function the liboffload API function to be called.
/// \param Args the arguments to be passed to the liboffload API function.
/// \throw sycl::exception with sycl::errc::runtime if the call was not
/// successful.
template <typename FunctionType, typename... ArgsT>
void callAndThrow(ContextImpl &Context, FunctionType &Function,
                  ArgsT &&...Args) {
  auto Err = callNoCheck(Function, std::forward<ArgsT>(Args)...);
  checkAndThrow(Context, Err);
}

/// Converts a liboffload backend to the matching SYCL backend.
///
/// \param Backend is the liboffload backend to convert.
/// \return the matching SYCL backend.
backend convertBackend(ol_platform_backend_t Backend);

/// Converts a SYCL device type to the matching liboffload device type.
///
/// \param DeviceType is the SYCL device type to convert.
/// \return the matching liboffload device type.
ol_device_type_t convertDeviceTypeToOL(info::device_type DeviceType);

/// Converts a liboffload device type to the matching SYCL device type.
///
/// \param DeviceType is the liboffload device type to convert.
/// \return the matching SYCL device type.
info::device_type convertDeviceTypeToSYCL(ol_device_type_t DeviceType);

/// Converts a SYCL USM kind to the matching liboffload allocation type.
///
/// \param USMKind is the SYCL USM kind to convert.
/// \return the matching liboffload allocation type.
ol_alloc_type_t getOlAllocType(usm::alloc USMKind);

/// Helper for static assertions in the discarded branch of an if constexpr.
template <typename> inline constexpr bool AlwaysFalse = false;

/// Helper to map SYCL information descriptors to `OL_<HANDLE>_INFO_<SMTH>`.
///
/// Typical usage:
/// \code
///   using Map = InfoOLMapping<ol_foo_info_t>;
///   constexpr auto OLInfo = mapInfoDesc<FromDesc, ol_foo_info_t>(
///       Map::M<DescVal0>{OL_FOO_INFO_VAL0},
///       Map::M<DescVal1>{OL_FOO_INFO_VAL1},
///       ...)
/// \endcode
template <typename To> struct InfoOLMapping {
  template <typename From> struct M {
    To Value;
    constexpr M(To Val) : Value(Val) {}
  };
};
template <typename From, typename To, typename... Ts>
constexpr To
mapInfoDesc(typename InfoOLMapping<To>::template M<Ts>... Mappings) {
  return std::get<typename InfoOLMapping<To>::template M<From>>(
             std::tuple{Mappings...})
      .Value;
}

/// Converts a UnifiedRangeView into the liboffload
/// ol_kernel_launch_size_args_t format.
ol_kernel_launch_size_args_t convertToOlRange(const UnifiedRangeView &Range);

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_SRC_DETAIL_OFFLOAD_OFFLOAD_UTILS_HPP
