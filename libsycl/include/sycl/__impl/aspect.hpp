//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_ASPECT_HPP
#define _LIBSYCL___IMPL_ASPECT_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstdint>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// SYCL 2020 4.6.4.5. Aspects.
enum class aspect : std::uint32_t {
  cpu,
  gpu,
  accelerator,
  custom,
  emulated,
  host_debuggable,
  fp16,
  fp64,
  atomic64,
  image,
  online_compiler,
  online_linker,
  queue_profiling,
  usm_device_allocations,
  usm_host_allocations,
  usm_atomic_host_allocations,
  usm_shared_allocations,
  usm_atomic_shared_allocations,
  usm_system_allocations
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_ASPECT_HPP
