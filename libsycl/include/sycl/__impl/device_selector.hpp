//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the standard device selectors
/// (SYCL 2020 4.6.1.1. Device selector).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DEVICE_SELECTOR_HPP
#define _LIBSYCL___IMPL_DEVICE_SELECTOR_HPP

#include <sycl/__impl/aspect.hpp>
#include <sycl/__impl/detail/config.hpp>

#include <functional>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class device;

namespace detail {

// SYCL 2020 4.6.1.1. Device selector:
// The interface for a device selector is any object that meets the C++ named
// requirement Callable, taking a parameter of type const device & and returning
// a value that is implicitly convertible to int.
using DeviceSelectorInvocableType = std::function<int(const sycl::device &)>;

template <typename DeviceSelector>
using EnableIfDeviceSelectorIsInvocable = std::enable_if_t<
    std::is_invocable_r_v<int, DeviceSelector &, const device &>>;

/// Returns a SYCL device instance chosen by the device selector provided.
///
/// \param DeviceSelector is SYCL 2020 device selector, a simple callable that
/// takes a device and returns an int.
/// \return device chosen by selector.
_LIBSYCL_EXPORT device
SelectDevice(const DeviceSelectorInvocableType &DeviceSelector);

} // namespace detail

/// Standard device selector to select SYCL device from any supported SYCL
/// backend based on an implementation-defined heuristic.
///
/// \param Dev device to calculate the score for.
/// \return score value for the provided device. Further device selection is
/// based on score values.
_LIBSYCL_EXPORT int default_selector_v(const device &Dev);

/// Standard device selector to select SYCL device from any supported SYCL
/// backend for which the device type is info::device_type::gpu.
///
/// \param Dev device to calculate the score for.
/// \return score value for the provided device. Further device selection is
/// based on score values.
_LIBSYCL_EXPORT int gpu_selector_v(const device &Dev);

/// Standard device selector to select SYCL device from any supported SYCL
/// backend for which the device type is info::device_type::cpu.
///
/// \param Dev device to calculate the score for.
/// \return score value for the provided device. Further device selection is
/// based on score values.
_LIBSYCL_EXPORT int cpu_selector_v(const device &Dev);

/// Standard device selector to select SYCL device from any supported SYCL
/// backend for which the device type is info::device_type::accelerator.
///
/// \param Dev device to calculate the score for.
/// \return score value for the provided device. Further device selection is
/// based on score values.
_LIBSYCL_EXPORT int accelerator_selector_v(const device &Dev);

/// Returns a selector object that selects a SYCL device from any supported SYCL
/// backend which contains all the requested aspects.
///
/// \param RequireList requested aspects,  i.e. for the specific device dev and
/// each aspect devAspect from RequireList dev.has(devAspect) equals true.
/// \param DenyList all the aspects that have to be avoided, i.e. for the
/// specific device dev and each aspect devAspect from denyList
/// dev.has(devAspect) equals false.
/// \return a selector object
_LIBSYCL_EXPORT detail::DeviceSelectorInvocableType
aspect_selector(const std::vector<aspect> &RequireList,
                const std::vector<aspect> &DenyList = {});

/// Returns a selector object that selects a SYCL device from any supported SYCL
/// backend which contains all the requested aspects.
///
/// \param AspectList requested aspects,  i.e. for the specific device dev and
/// each aspect devAspect from AspectList dev.has(devAspect) equals true.
/// \return a selector object
template <typename... AspectListT>
detail::DeviceSelectorInvocableType aspect_selector(AspectListT... AspectList) {
  std::vector<aspect> RequireList;
  RequireList.reserve(sizeof...(AspectList));
  (RequireList.emplace_back(AspectList), ...);

  return aspect_selector(RequireList, {});
}

/// Returns a selector object that selects a SYCL device from any supported SYCL
/// backend which contains all the requested aspects.
///
/// \param AspectList requested aspects,  i.e. for the specific device dev and
/// each aspect devAspect from AspectList dev.has(devAspect) equals true.
/// \return a selector object
template <aspect... AspectList>
detail::DeviceSelectorInvocableType aspect_selector() {
  return aspect_selector({AspectList...}, {});
}

_LIBSYCL_END_NAMESPACE_SYCL

#endif //_LIBSYCL___IMPL_DEVICE_SELECTOR_HPP
