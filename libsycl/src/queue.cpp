//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/queue.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

queue::queue(const device &syclDevice, const async_handler &asyncHandler,
             const property_list &propList) {
  impl = detail::QueueImpl::create(*detail::getSyclObjImpl(syclDevice),
                                   asyncHandler, propList);
}

backend queue::get_backend() const noexcept { return impl->getBackend(); }

context queue::get_context() const {
  return detail::createSyclObjFromImpl<context>(impl->getContext());
}

device queue::get_device() const {
  return detail::createSyclObjFromImpl<device>(impl->getDevice());
}

bool queue::is_in_order() const { return impl->isInOrder(); }

_LIBSYCL_END_NAMESPACE_SYCL
