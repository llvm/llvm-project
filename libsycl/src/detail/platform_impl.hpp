//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_PLATFORM_IMPL
#define _LIBSYCL_PLATFORM_IMPL

#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/platform.hpp>

#include <detail/common.hpp>
#include <detail/offload/info_code.hpp>
#include <detail/offload/offload_utils.hpp>

#include <OffloadAPI.h>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class platform_impl {
public:
  /// Constructs platform_impl from a platform handle.
  ///
  /// \param Platform is a raw offload library handle representing platform.
  /// \param PlatformIndex is a platform index in a backend (needed for a proper
  /// indexing in device selector).
  //
  // Platforms can only be created under `GlobalHandler`'s ownership via
  // `platform_impl::getOrMakePlatformImpl` method.
  explicit platform_impl(ol_platform_handle_t Platform, size_t PlatformIndex);

  ~platform_impl() = default;

  /// Returns the backend associated with this platform.
  backend getBackend() const noexcept { return MBackend; }

  /// Returns range-view to all SYCL platforms from all backends that are
  /// available in the system.
  static range_view<std::unique_ptr<platform_impl>> getPlatforms();

  /// Returns raw underlying offload platform handle.
  ///
  /// It does not retain handle. It is caller responsibility to make sure that
  /// platform stays alive while raw handle is in use.
  ///
  /// \return a raw plug-in platform handle.
  const ol_platform_handle_t &getHandleRef() const { return MOffloadPlatform; }

  /// Returns platform index in a backend (needed for a proper indexing in
  /// device selector).
  size_t getPlatformIndex() const { return MOffloadPlatformIndex; }

  /// Queries the cache to see if the specified offloading RT platform has been
  /// seen before.  If so, return the cached platform_impl, otherwise create a
  /// new one and cache it.
  ///
  /// \param Platform is the offloading RT Platform handle representing the
  /// platform
  /// \param PlatformIndex is a platform index in a backend (needed for a proper
  /// indexing in device selector).
  /// \return the platform_impl representing the offloading RT platform
  static platform_impl *getPlatformImpl(ol_platform_handle_t Platform);

  /// Queries this SYCL platform for info.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const {
    // for now we have only std::string properties
    static_assert(std::is_same_v<typename Param::return_type, std::string>);
    size_t ExpectedSize = 0;
    call_and_throw(olGetPlatformInfoSize, MOffloadPlatform,
                   detail::OffloadInfoCode<Param>::value, &ExpectedSize);
    std::string Result;
    Result.resize(ExpectedSize - 1);
    call_and_throw(olGetPlatformInfo, MOffloadPlatform,
                   detail::OffloadInfoCode<Param>::value, ExpectedSize,
                   Result.data());
    return Result;
  }

private:
  ol_platform_handle_t MOffloadPlatform{};
  size_t MOffloadPlatformIndex{};
  ol_platform_backend_t MOffloadBackend{OL_PLATFORM_BACKEND_UNKNOWN};
  backend MBackend{};
};

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_PLATFORM_IMPL
