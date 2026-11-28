//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 local_accessor class
/// (4.7.6.11.), which allocates work-group local memory for a kernel.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_LOCAL_ACCESSOR_HPP
#define _LIBSYCL___IMPL_LOCAL_ACCESSOR_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/local_accessor_base.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/property_list.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class handler;

/// Allocates device local memory and provides access to this memory from
/// within a SYCL kernel function.
///
/// The local memory that is allocated is shared between all work-items of a
/// work-group. A local accessor can only be used in a kernel invoked with an
/// nd_range; submitting it with single_task or with a parallel_for taking a
/// range throws sycl::exception with sycl::errc::kernel_argument.
///
/// \sa SYCL 2020 4.7.6.11. Local accessor.
template <typename DataT, int Dimensions = 1>
class _LIBSYCL_REIFIED_KERNEL_PARAMETER local_accessor
    : public detail::LocalAccessorBase<DataT, Dimensions> {
  using Base = detail::LocalAccessorBase<DataT, Dimensions>;

public:
  using value_type = typename Base::value_type;
  using size_type = typename Base::size_type;

  /// Constructs an empty local accessor.
  ///
  /// A default constructed local accessor can be passed to a SYCL kernel
  /// function, but attempting to access data elements from it produces
  /// undefined behavior.
  local_accessor() = default;

  /// Constructs a local_accessor for accessing local memory of a single DataT
  /// element within a SYCL kernel function on the queue associated with
  /// \p commandGroupHandlerRef.
  ///
  /// Available only when (Dimensions == 0).
  ///
  /// \param commandGroupHandlerRef is the handler of the command group that
  /// uses the accessor.
  /// \param propList is a list of properties for accessor construction.
  /// \throw sycl::exception with sycl::errc::invalid if \p propList contains a
  /// property that cannot be applied to a local accessor.
  template <int Dims = Dimensions, std::enable_if_t<Dims == 0, bool> = true>
  local_accessor([[maybe_unused]] handler &commandGroupHandlerRef,
                 const property_list &propList = {})
      : Base(range<1>{1}, propList) {}

  /// Constructs a local_accessor for accessing local memory of an array of
  /// DataT elements within a SYCL kernel function on the queue associated
  /// with \p commandGroupHandlerRef.
  ///
  /// Available only when (Dimensions > 0).
  ///
  /// \param allocationSize is the number of elements in the array.
  /// \param commandGroupHandlerRef is the handler of the command group that
  /// uses the accessor.
  /// \param propList is a list of properties for accessor construction.
  /// \throw sycl::exception with sycl::errc::invalid if \p propList contains a
  /// property that cannot be applied to a local accessor.
  template <int Dims = Dimensions, std::enable_if_t<(Dims > 0), bool> = true>
  local_accessor(range<Dimensions> allocationSize,
                 [[maybe_unused]] handler &commandGroupHandlerRef,
                 const property_list &propList = {})
      : Base(allocationSize, propList) {}

  /// Converts a read-write local accessor to a read-only one that accesses the
  /// same local memory.
  template <typename OtherDataT,
            std::enable_if_t<
                std::is_const_v<DataT> &&
                    std::is_same_v<OtherDataT, std::remove_const_t<DataT>>,
                bool> = true>
  local_accessor(const local_accessor<OtherDataT, Dimensions> &Other) noexcept
      : Base(Other) {}

  friend bool operator==(const local_accessor &Lhs, const local_accessor &Rhs) {
    return Lhs.isEqual(Rhs);
  }

  friend bool operator!=(const local_accessor &Lhs, const local_accessor &Rhs) {
    return !(Lhs == Rhs);
  }

  /// Swaps the contents of the current accessor with the contents of
  /// \p other.
  void swap(local_accessor &other) { std::swap(*this, other); }

  /// Assigns \p other to the single element that is accessed by this accessor.
  ///
  /// Available only when DataT is not const qualified and (Dimensions == 0).
  /// Must be called within a command.
  ///
  /// \param other is the value to assign.
  /// \return this accessor.
  template <int Dims = Dimensions,
            std::enable_if_t<Dims == 0 && !std::is_const_v<DataT>, bool> = true>
  const local_accessor &operator=(const value_type &other) const {
    *this->getPtr() = other;
    return *this;
  }

  /// \overload
  ///
  /// \param other is the value to assign.
  template <int Dims = Dimensions,
            std::enable_if_t<Dims == 0 && !std::is_const_v<DataT>, bool> = true>
  const local_accessor &operator=(value_type &&other) const {
    *this->getPtr() = std::move(other);
    return *this;
  }

private:
  friend struct std::hash<local_accessor>;
};

_LIBSYCL_END_NAMESPACE_SYCL

template <typename DataT, int Dimensions>
struct std::hash<sycl::local_accessor<DataT, Dimensions>> {
  std::size_t
  operator()(const sycl::local_accessor<DataT, Dimensions> &Acc) const {
    return std::hash<std::uint64_t>{}(Acc.getId());
  }
};

#endif // _LIBSYCL___IMPL_LOCAL_ACCESSOR_HPP
