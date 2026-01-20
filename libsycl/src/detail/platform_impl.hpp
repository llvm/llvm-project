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

#include <detail/offload/offload_utils.hpp>

#include <OffloadAPI.h>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

using PlatformImplUPtr = std::unique_ptr<PlatformImpl>;

class PlatformImpl {
  struct PrivateTag {
    explicit PrivateTag() = default;
  };

public:
  /// Constructs PlatformImpl from a platform handle.
  ///
  /// \param Platform is a raw offload library handle representing platform.
  /// \param PlatformIndex is a platform index in a backend (needed for a proper
  /// indexing in device selector).
  /// All platform impls are created during first getPlatforms() call.
  PlatformImpl(ol_platform_handle_t Platform, size_t PlatformIndex, PrivateTag);

  ~PlatformImpl() = default;

  /// Returns the backend associated with this platform.
  ///
  /// \returns sycl::backend associated with this platform.
  backend getBackend() const noexcept { return MBackend; }

  /// Returns all SYCL platforms from all backends that are
  /// available in the system.
  ///
  /// \returns std::vector of all platforms that are available in the system.
  static const std::vector<PlatformImplUPtr> &getPlatforms();

  /// Returns the raw underlying offload platform handle.
  ///
  /// The caller is responsible for ensuring that the returned handle is only
  /// used while the PlatformImpl object from which it was obtained is still
  /// within its lifetime.
  ///
  /// \return a raw offload platform handle.
  const ol_platform_handle_t &getHandleRef() const { return MOffloadPlatform; }

  /// Queries the cache to get the implementation for specified offloading RT
  /// platform. All platform implementation objects are created at first
  /// get_platforms call.
  ///
  /// \param Platform is the offloading RT Platform handle representing the
  /// platform.
  /// \return the PlatformImpl representing the offloading RT platform.
  static PlatformImpl &getPlatformImpl(ol_platform_handle_t Platform);

  /// Queries this platform for info.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type getInfo() const {
    // For now we have only std::string properties
    static_assert(std::is_same_v<typename Param::return_type, std::string>);

    using namespace info::platform;
    using Map = info_ol_mapping<ol_platform_info_t>;

    constexpr ol_platform_info_t olInfo =
        map_info_desc<Param, ol_platform_info_t>(
            Map::M<version>{OL_PLATFORM_INFO_VERSION},
            Map::M<name>{OL_PLATFORM_INFO_NAME},
            Map::M<vendor>{OL_PLATFORM_INFO_VENDOR_NAME});

    size_t ExpectedSize = 0;
    callAndThrow(olGetPlatformInfoSize, MOffloadPlatform, olInfo,
                 &ExpectedSize);
    std::string Result;
    Result.resize(ExpectedSize - 1);
    callAndThrow(olGetPlatformInfo, MOffloadPlatform, olInfo, ExpectedSize,
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
