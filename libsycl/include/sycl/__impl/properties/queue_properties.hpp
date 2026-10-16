//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the SYCL queue properties (4.6.5.1.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_PROPERTIES_QUEUE_PROPERTIES_HPP
#define _LIBSYCL___IMPL_PROPERTIES_QUEUE_PROPERTIES_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/property_helpers.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace property::queue {

// Declared as struct to match the class synopsis in SYCL 2020 4.6.5.5., so
// that a user redeclaring one of these does not trip -Wmismatched-tags.

/// Requires the queue to execute its command groups in the order in which they
/// were submitted.
struct in_order
    : public detail::property_of<in_order, detail::PropKind::queue_in_order,
                                 sycl::queue> {};

/// Requires the queue to capture profiling information for the command groups
/// submitted to it. Requires aspect::queue_profiling on the queue's device.
struct enable_profiling
    : public detail::property_of<enable_profiling,
                                 detail::PropKind::queue_enable_profiling,
                                 sycl::queue> {};

} // namespace property::queue

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_PROPERTIES_QUEUE_PROPERTIES_HPP
