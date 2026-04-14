//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/event.hpp>

#include <detail/event_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

backend event::get_backend() const noexcept { return impl->getBackend(); }

void event::wait(const std::vector<event> &EventList) {
  for (auto Event : EventList) {
    Event.wait();
  }
}

void event::wait() { impl->wait(); }

_LIBSYCL_END_NAMESPACE_SYCL
