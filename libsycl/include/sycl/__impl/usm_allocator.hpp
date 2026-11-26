//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL USM allocator
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_USM_ALLOCATOR_HPP
#define _LIBSYCL___IMPL_USM_ALLOCATOR_HPP

#include <sycl/__impl/aspect.hpp>
#include <sycl/__impl/context.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/property_list.hpp>
#include <sycl/__impl/queue.hpp>
#include <sycl/__impl/usm_alloc_type.hpp>
#include <sycl/__impl/usm_functions.hpp>

#include <sycl/__impl/detail/property_interface.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

/// SYCL 2020 4.8.3.1. C++ allocator interface.
template <typename T, usm::alloc AllocKind, std::size_t Alignment = 0>
class usm_allocator : private detail::PropertyInterface<
                          usm_allocator<T, AllocKind, Alignment>> {
  static_assert(
      AllocKind != usm::alloc::device,
      "usm_allocator does not support AllocKind == usm::alloc::device");

public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

public:
  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind, Alignment> other;
  };

  usm_allocator() = delete;

  /// Constructs an allocator that allocates USM of kind AllocKind on
  /// \p syclDevice.
  ///
  /// \param syclContext a context containing syclDevice or its parent device if
  /// syclDevice is a subdevice.
  /// \param syclDevice the device to use for the allocations.
  /// \param propList the list of properties for the allocations.
  /// \throw sycl::exception with sycl::errc::feature_not_supported if AllocKind
  /// is usm::alloc::host and no device in syclContext has
  /// aspect::usm_host_allocations.
  /// \throw sycl::exception with sycl::errc::invalid if AllocKind is
  /// usm::alloc::shared and syclContext does not contain syclDevice.
  /// \throw sycl::exception with sycl::errc::feature_not_supported if AllocKind
  /// is usm::alloc::shared and syclDevice does not have
  /// aspect::usm_shared_allocations.
  usm_allocator(const context &syclContext, const device &syclDevice,
                const property_list &propList = {})
      : MContext(syclContext), MDevice(syclDevice), MPropList(propList) {
    std::vector<device> ContextDevices = syclContext.get_devices();
    if constexpr (AllocKind == usm::alloc::host) {
      if (std::none_of(ContextDevices.begin(), ContextDevices.end(),
                       [](const device &Dev) {
                         return Dev.has(aspect::usm_host_allocations);
                       }))
        throw exception(
            syclContext, make_error_code(errc::feature_not_supported),
            "None of the context's devices support host USM allocations.");
    } else {
      if (std::find(ContextDevices.begin(), ContextDevices.end(), syclDevice) ==
          ContextDevices.end())
        throw exception(
            syclContext, make_error_code(errc::invalid),
            "Specified device is not contained by specified context.");
      if (!syclDevice.has(aspect::usm_shared_allocations))
        throw exception(syclContext,
                        make_error_code(errc::feature_not_supported),
                        "Device doesn't support shared USM allocations.");
    }
  }

  /// \overload
  ///
  /// \param syclQueue the queue that provides the context and device.
  /// \param propList the list of properties for the allocations.
  usm_allocator(const queue &syclQueue, const property_list &propList = {})
      : usm_allocator(syclQueue.get_context(), syclQueue.get_device(),
                      propList) {}

  /// Constructs an allocator that uses the same context, device and properties
  /// as \p other.
  template <typename U>
  usm_allocator(const usm_allocator<U, AllocKind, Alignment> &other) noexcept
      : MContext(other.MContext), MDevice(other.MDevice),
        MPropList(other.MPropList) {}

  usm_allocator(const usm_allocator &other) = default;
  usm_allocator(usm_allocator &&other) noexcept = default;
  ~usm_allocator() = default;

  usm_allocator &operator=(const usm_allocator &other) = default;
  usm_allocator &operator=(usm_allocator &&other) noexcept = default;

  /// Allocates USM for \p count elements of type T.
  ///
  /// \param count the number of elements to allocate.
  /// \return a pointer to the newly allocated memory, or nullptr if \p count
  /// is 0.
  /// \throw a sycl::exception with sycl::errc::memory_allocation if the
  /// allocation fails.
  value_type *allocate(std::size_t count) {
    if (count == 0)
      return nullptr;

    auto Result = aligned_alloc<value_type>(Alignment, count, MDevice, MContext,
                                            AllocKind, MPropList);

    if (!Result)
      throw exception(MContext, make_error_code(errc::memory_allocation),
                      "USM allocation failed");

    return Result;
  }

  /// Deallocates memory previously returned by allocate().
  ///
  /// \param Ptr a pointer returned by allocate(), or a null pointer.
  /// \param count the number of elements Ptr was allocated for.
  void deallocate(value_type *Ptr, [[maybe_unused]] std::size_t count) {
    if (Ptr)
      sycl::free(Ptr, MContext);
  }

  // SYCL 2020 4.5.4.1. The property interface.
  using detail::PropertyInterface<usm_allocator>::has_property;
  using detail::PropertyInterface<usm_allocator>::get_property;

  /// \return true if \p lhs and \p rhs allocate the same kind of USM in the
  /// same context and on the same device.
  template <class U, usm::alloc AllocKindU, std::size_t AlignmentU>
  friend bool operator==(const usm_allocator<T, AllocKind, Alignment> &lhs,
                         const usm_allocator<U, AllocKindU, AlignmentU> &rhs) {
    return AllocKind == AllocKindU && lhs.MContext == rhs.MContext &&
           lhs.MDevice == rhs.MDevice;
  }

  template <class U, usm::alloc AllocKindU, std::size_t AlignmentU>
  friend bool operator!=(const usm_allocator<T, AllocKind, Alignment> &lhs,
                         const usm_allocator<U, AllocKindU, AlignmentU> &rhs) {
    return !(lhs == rhs);
  }

private:
  const property_list &getPropertyList() const noexcept { return MPropList; }

  template <typename, usm::alloc, std::size_t> friend class usm_allocator;
  friend class detail::PropertyInterface<usm_allocator>;

  context MContext;
  device MDevice;
  property_list MPropList;
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif
