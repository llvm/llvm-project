//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the ContextImpl class, which
/// implements sycl::context functionality.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_CONTEXT_IMPL
#define _LIBSYCL_CONTEXT_IMPL

#include <sycl/__impl/async_handler.hpp>
#include <sycl/__impl/context.hpp>
#include <sycl/__impl/detail/config.hpp>

#include <OffloadAPI.h>

#include <functional>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class property_list;

namespace detail {

class PlatformImpl;
class DeviceImpl;

/// Context represents the runtime data structures and state required by a SYCL
/// backend API to interact with a group of devices associated with a platform.
class ContextImpl : public std::enable_shared_from_this<ContextImpl> {
  struct Private {
    explicit Private() = default;
  };

public:
  /// Constructs a context implementation for the provided devices.
  ///
  /// \param DeviceList is the list of devices associated with this context.
  /// \param AsyncHandler is a SYCL asynchronous exception handler.
  /// \param PropList is a list of context properties.
  ContextImpl(std::vector<DeviceImpl *> &&DeviceList,
              const async_handler &AsyncHandler, const property_list &PropList,
              Private);

  /// Releases the underlying offload context handle.
  ~ContextImpl();

  /// Gets asynchronous exception handler.
  ///
  /// \return an instance of SYCL async_handler.
  const async_handler &get_async_handler() const { return MAsyncHandler; }

  /// Constructs a ContextImpl with a provided arguments. Variadic helper.
  /// Restrics ways of ContextImpl creation.
  template <typename... Ts>
  static std::shared_ptr<ContextImpl> create(Ts &&...args) {
    return std::make_shared<ContextImpl>(std::forward<Ts>(args)..., Private{});
  }

  /// Returns the raw underlying offload context handle.
  ///
  /// The caller is responsible for ensuring that the returned handle is only
  /// used while this ContextImpl object is alive.
  ///
  /// \return the raw offload context handle.
  const ol_context_handle_t &getOLHandleRef() const { return MOffloadContext; }

  /// \return the platform this context is associated with.
  PlatformImpl &getPlatformImpl() const;

  /// Calls "callback" with every device associated
  /// with this context.
  void iterateDevices(const std::function<void(DeviceImpl *)> &callback) const;

  /// \return backend of the platform this context is associated with.
  backend getBackend() const;

private:
  const async_handler MAsyncHandler;
  const std::vector<DeviceImpl *> MDevices;
  ol_context_handle_t MOffloadContext{};
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_CONTEXT_IMPL
