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

void queue::wait() { impl->wait(); }

event queue::memcpy(void *dest, const void *src, std::size_t numBytes,
                    const std::vector<event> &depEvents) {
  std::shared_ptr<detail::EventImpl> EventImplPtr =
      impl->memcpy(dest, src, numBytes, detail::getSyclObjImpls(depEvents));
  assert(EventImplPtr);
  return detail::createSyclObjFromImpl<event>(EventImplPtr);
}

event queue::getLastEvent() {
  return detail::createSyclObjFromImpl<event>(impl->getLastEvent());
}

void queue::setKernelParameters(const std::vector<event> &Events,
                                const detail::UnifiedRangeView &Range) {
  return impl->setKernelParameters(detail::getSyclObjImpls(Events), Range);
}

void queue::submitKernelImpl(detail::DeviceKernelInfo &KernelInfo,
                             void *ArgData, size_t ArgSize) {
  impl->submitKernelImpl(KernelInfo, ArgData, ArgSize);
}

_LIBSYCL_END_NAMESPACE_SYCL
