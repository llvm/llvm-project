//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/offload/offload_utils.hpp>

#include <cassert>
#include <limits>

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
    throw exception(make_error_code(errc::runtime), "Unsupported backend");
  }
}

ol_device_type_t convertDeviceTypeToOL(info::device_type DeviceType) {
  switch (DeviceType) {
  case info::device_type::all:
    return OL_DEVICE_TYPE_ALL;
  case info::device_type::gpu:
    return OL_DEVICE_TYPE_GPU;
  case info::device_type::cpu:
    return OL_DEVICE_TYPE_CPU;
  case info::device_type::automatic:
    return OL_DEVICE_TYPE_DEFAULT;
  default:
    throw exception(sycl::make_error_code(sycl::errc::runtime),
                    "Device type is not supported");
  }
}

info::device_type convertDeviceTypeToSYCL(ol_device_type_t DeviceType) {
  switch (DeviceType) {
  case OL_DEVICE_TYPE_GPU:
    return info::device_type::gpu;
  case OL_DEVICE_TYPE_CPU:
    return info::device_type::cpu;
  default:
    throw exception(sycl::make_error_code(sycl::errc::runtime),
                    "Device type is not supported");
  }
}

ol_alloc_type_t getOlAllocType(usm::alloc USMKind) {
  switch (USMKind) {
  case usm::alloc::host:
    return OL_ALLOC_TYPE_HOST;
  case usm::alloc::device:
    return OL_ALLOC_TYPE_DEVICE;
  case usm::alloc::shared:
    return OL_ALLOC_TYPE_MANAGED;
  case usm::alloc::unknown:
    // usm::alloc::unknown can be returned to user from get_pointer_type but it
    // can't be converted to a valid backend type.
    throw exception(sycl::make_error_code(sycl::errc::runtime),
                    "USM kind is not supported");
  }
}

ol_kernel_launch_size_args_t convertToOlRange(const UnifiedRangeView &Range) {
  assert(Range.MDims < 4 && "Invalid dimensions.");

  uint32_t GlobalSize[3] = {1, 1, 1};
  if (Range.MGlobalSize) {
    for (size_t I = 0; I < Range.MDims; ++I) {
      assert(Range.MGlobalSize[I] <= std::numeric_limits<uint32_t>::max());
      GlobalSize[I] = static_cast<uint32_t>(Range.MGlobalSize[I]);
    }
  }

  uint32_t GroupSize[3] = {1, 1, 1};
  if (Range.MLocalSize) {
    for (size_t I = 0; I < Range.MDims; ++I) {
      assert(Range.MLocalSize[I] <= std::numeric_limits<uint32_t>::max() &&
             Range.MLocalSize[I] != 0);
      GroupSize[I] = static_cast<uint32_t>(Range.MLocalSize[I]);
    }
  }

  // We have the following mapping between dimensions with SPIR-V builtins:
  // 1D: id[0] -> x
  // 2D: id[0] -> y, id[1] -> x
  // 3D: id[0] -> z, id[1] -> y, id[2] -> x
  // So in order to ensure the correctness we update all the kernel
  // parameters accordingly.
  if (Range.MDims > 1) {
    // TODO: Offset is not yet supported in liboffload, so ignore it for now.
    std::swap(GlobalSize[0], GlobalSize[Range.MDims - 1]);
    std::swap(GroupSize[0], GroupSize[Range.MDims - 1]);
  }

  ol_kernel_launch_size_args_t olRange = {};
  olRange.Dimensions = Range.MDims;
  olRange.NumGroups.x = GlobalSize[0] / GroupSize[0];
  olRange.NumGroups.y = GlobalSize[1] / GroupSize[1];
  olRange.NumGroups.z = GlobalSize[2] / GroupSize[2];
  olRange.GroupSize.x = GroupSize[0];
  olRange.GroupSize.y = GroupSize[1];
  olRange.GroupSize.z = GroupSize[2];
  olRange.DynSharedMemory = 0;
  return olRange;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
