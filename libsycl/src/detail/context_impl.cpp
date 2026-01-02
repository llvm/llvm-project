//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/platform_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

void ContextImpl::iterateDevices(
    const std::function<void(DeviceImpl *)> &callback) const {
  // Intentionally don't store devices in context now. This class should be
  // reimplemented once liboffload adds context support. Treat context as
  // default context that is associated with all devices in the platform.
  return MPlatform.iterateDevices(info::device_type::all, callback);
}

backend ContextImpl::getBackend() const { return MPlatform.getBackend(); }

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
