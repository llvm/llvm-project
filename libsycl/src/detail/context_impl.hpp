//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_CONTEXT_IMPL
#define _LIBSYCL_CONTEXT_IMPL

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/detail/config.hpp>

#include <OffloadAPI.h>

#include <functional>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class PlatformImpl;
class DeviceImpl;

/// Context dummy (w/o liboffload handle) that represents all devices
/// in platform.
///
/// Presence of context object is essential for many APIs. This dummy is a way
/// to support them in case of absence of context support in liboffload. For
/// backends where context exists and participates in operations liboffload
/// plugins create and use default context that represents all devices in that
/// platform. Duplicating this logic here.
class ContextImpl : public std::enable_shared_from_this<ContextImpl> {
  struct Private {
    explicit Private() = default;
  };

public:
  /// Constructs a ContextImpl using a platform.
  ///
  /// Newly created instance represents all devices in platform.
  ///
  /// \param Platform is a platform to associate this context with.
  ContextImpl(PlatformImpl &Platform, Private) : MPlatform(Platform) {}

  /// Constructs a ContextImpl with a provided arguments. Variadic helper.
  /// Restrics ways of ContextImpl creation.
  template <typename... Ts>
  static std::shared_ptr<ContextImpl> create(Ts &&...args) {
    return std::make_shared<ContextImpl>(std::forward<Ts>(args)..., Private{});
  }

  /// Returns associated platform
  ///
  /// \return platform implementation object this context is associated with.
  PlatformImpl &getPlatformImpl() const { return MPlatform; }

  /// Calls "callback" with every device associated
  /// with this context.
  void iterateDevices(const std::function<void(DeviceImpl *)> &callback) const;

  /// Returns backend of the platform this context is associated with.
  ///
  /// \return SYCL backend.
  backend getBackend() const;

private:
  PlatformImpl &MPlatform;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_CONTEXT_IMPL
