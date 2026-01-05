//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/offload/offload_utils.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

const char *stringifyErrorCode(ol_errc_t Error) {
  switch (Error) {
#define _OFFLOAD_ERRC(NAME)                                                    \
  case NAME:                                                                   \
    return #NAME;
    _OFFLOAD_ERRC(OL_ERRC_UNKNOWN)
    _OFFLOAD_ERRC(OL_ERRC_HOST_IO)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_BINARY)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_NULL_POINTER)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_ARGUMENT)
    _OFFLOAD_ERRC(OL_ERRC_NOT_FOUND)
    _OFFLOAD_ERRC(OL_ERRC_OUT_OF_RESOURCES)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_SIZE)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_ENUMERATION)
    _OFFLOAD_ERRC(OL_ERRC_HOST_TOOL_NOT_FOUND)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_VALUE)
    _OFFLOAD_ERRC(OL_ERRC_UNIMPLEMENTED)
    _OFFLOAD_ERRC(OL_ERRC_UNSUPPORTED)
    _OFFLOAD_ERRC(OL_ERRC_ASSEMBLE_FAILURE)
    _OFFLOAD_ERRC(OL_ERRC_COMPILE_FAILURE)
    _OFFLOAD_ERRC(OL_ERRC_LINK_FAILURE)
    _OFFLOAD_ERRC(OL_ERRC_BACKEND_FAILURE)
    _OFFLOAD_ERRC(OL_ERRC_UNINITIALIZED)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_NULL_HANDLE)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_PLATFORM)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_DEVICE)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_QUEUE)
    _OFFLOAD_ERRC(OL_ERRC_INVALID_EVENT)
    _OFFLOAD_ERRC(OL_ERRC_SYMBOL_KIND)
#undef _OFFLOAD_ERRC

  default:
    return "Unknown error code";
  }
}

backend convertBackend(ol_platform_backend_t Backend) {
  switch (Backend) {
  case OL_PLATFORM_BACKEND_LEVEL_ZERO:
    return backend::level_zero;
  case OL_PLATFORM_BACKEND_CUDA:
    return backend::cuda;
  case OL_PLATFORM_BACKEND_AMDGPU:
    return backend::hip;
  default:
    throw exception(make_error_code(errc::runtime),
                    "convertBackend: Unsupported backend");
  }
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
