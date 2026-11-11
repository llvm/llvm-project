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

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/property_list.hpp>
#include <sycl/__impl/queue.hpp>
#include <sycl/__impl/usm_alloc_type.hpp>
#include <sycl/__impl/usm_functions.hpp>

#include <sycl/__impl/detail/property_interface.hpp>

#include <cstddef>
#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

/// USM Allocator
/// \sa 4.8.3.1. C++ allocator interface
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
  usm_allocator(const context &syclContext, const device &syclDevice,
                const property_list &propList = {})
      : MContext(syclContext), MDevice(syclDevice), MPropList(propList) {}
  usm_allocator(const queue &syclQueue, const property_list &propList = {})
      : usm_allocator(syclQueue.get_context(), syclQueue.get_device(),
                      propList) {}

  template <typename U>
  usm_allocator(const usm_allocator<U, AllocKind, Alignment> &other) noexcept
      : MContext(other.MContext), MDevice(other.MDevice),
        MPropList(other.MPropList) {}

  usm_allocator(const usm_allocator &other) = default;
  usm_allocator(usm_allocator &&other) noexcept = default;
  ~usm_allocator() = default;

  usm_allocator &operator=(const usm_allocator &other) = default;
  usm_allocator &operator=(usm_allocator &&other) = default;

  /// Allocate \p count elements
  /// \p count number of elements to allocate memory for
  /// \return pointer to the allocated memory, or nullptr for \p count == 0
  /// \throw a sycl memory_allocation exception on failure
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

  /// Deallocate a USM pointer
  /// \p Ptr Address to be deallocated
  /// \p count of elements the address was allocated for
  void deallocate(value_type *Ptr, std::size_t count) {
    if (Ptr)
      free(Ptr, MContext);
  }

  // SYCL 2020 4.5.4.1. The property interface.
  using detail::PropertyInterface<usm_allocator>::has_property;
  using detail::PropertyInterface<usm_allocator>::get_property;

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
}; // usm_allocator

_LIBSYCL_END_NAMESPACE_SYCL

#endif
