//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/event.hpp>
#include <sycl/__impl/exception.hpp>

#include <detail/event_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

event::event() : impl(detail::EventImpl::createDefaultEvent()) {}

backend event::get_backend() const noexcept { return impl->getBackend(); }

void event::wait(const std::vector<event> &EventList) {
  for (auto Event : EventList) {
    Event.wait();
  }
}

void event::wait() { impl->wait(); }

void event::wait_and_throw() { impl->waitAndThrow(); }

void event::wait_and_throw(const std::vector<event> &EventList) {
  for (auto E : EventList) {
    E.wait_and_throw();
  }
}

std::vector<event> event::get_wait_list() {
  std::vector<event> Result;

  for (auto &EventImpl : impl->getWaitList())
    Result.push_back(detail::createSyclObjFromImpl<event>(EventImpl));

  return Result;
}

template <typename Param>
typename detail::is_event_profiling_info_desc_t<Param>
event::get_profiling_info() const {
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Profiling features are not supported.");
}

_LIBSYCL_END_NAMESPACE_SYCL
