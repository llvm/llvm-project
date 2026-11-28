//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/local_accessor_base.hpp>

#include <atomic>
#include <cstdint>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

std::uint64_t createLocalAccessorId() noexcept {
  // 0 is the identity of accessors constructed on device.
  static std::atomic<std::uint64_t> NextId{1};
  return NextId.fetch_add(1, std::memory_order_relaxed);
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL
