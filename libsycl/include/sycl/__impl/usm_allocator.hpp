//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL queue class, which
/// schedules kernels on a device.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_USM_ALLOCATOR_HPP
#define _LIBSYCL___IMPL_USM_ALLOCATOR_HPP

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/property_list.hpp>
#include <sycl/__impl/queue.hpp>
#include <sycl/__impl/usm_alloc_type.hpp>

#include <cstddef>
#include <type_traits>

namespace sycl {

// TODO: check allocKind and reject device
template <typename T, usm::alloc AllocKind, std::size_t Alignment = 0>
class usm_allocator {
public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

public:
  usm_allocator() = delete;
  usm_allocator(const context &syclContext, const device &syclDevice,
                const property_list &propList = {});
  usm_allocator(const queue &syclQueue, const property_list &propList = {});

  usm_allocator(const usm_allocator &other);
  usm_allocator(usm_allocator &&other) noexcept;
  ~usm_allocator() = default;

  usm_allocator &operator=(const usm_allocator &other);
  usm_allocator &
  operator=(usm_allocator &&other); // spec doesnt mark this as noexcept

  value_type *allocate(std::size_t count);

  void deallocate(value_type *Ptr, std::size_t count);

  template <class U, usm::alloc AllocKindU, std::size_t AlignmentU>
  friend bool operator==(const usm_allocator<T, AllocKind, Alignment> &lhs,
                         const usm_allocator<U, AllocKindU, AlignmentU> &rhs) {
    // TODO: implement this
    return true;
  }

  template <class U, usm::alloc AllocKindU, std::size_t AlignmentU>
  friend bool operator!=(const usm_allocator<T, AllocKind, Alignment> &lhs,
                         const usm_allocator<U, AllocKindU, AlignmentU> &rhs) {
    return !(lhs == rhs);
  }
}; // usm_allocator

} // namespace sycl

#endif
