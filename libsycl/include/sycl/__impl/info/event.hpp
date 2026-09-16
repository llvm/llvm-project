//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains SYCL 2020 event and event profiling info descriptors.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_INFO_EVENT_HPP
#define _LIBSYCL___IMPL_INFO_EVENT_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/info/desc_base.hpp>

#include <cstdint>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class event;

namespace detail {
/// Sentinel type used as the DescOf tag for event profiling info descriptors,
/// so they are distinct from regular event info descriptors.
struct event_profiling_tag {};
} // namespace detail

namespace info {
namespace event_profiling {
struct command_submit
    : detail::info_desc_tag<command_submit, detail::event_profiling_tag> {
  using return_type = std::uint64_t;
};

struct command_start
    : detail::info_desc_tag<command_start, detail::event_profiling_tag> {
  using return_type = std::uint64_t;
};

struct command_end
    : detail::info_desc_tag<command_end, detail::event_profiling_tag> {
  using return_type = std::uint64_t;
};
} // namespace event_profiling
} // namespace info

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INFO_EVENT_HPP
