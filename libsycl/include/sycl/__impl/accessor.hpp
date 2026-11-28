//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL accessor class (4.7.6.).
/// Only the specialization for target::local, deprecated in SYCL 2020
/// in favour of sycl::local_accessor, is defined so far.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_ACCESSOR_HPP
#define _LIBSYCL___IMPL_ACCESSOR_HPP

#include <sycl/__impl/access_enums.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/local_accessor_base.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/property_list.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class handler;

/// Provides access to data in a buffer from within a command.
///
/// \sa SYCL 2020 4.7.6.9. Buffer accessor for commands.
// TODO: only the target::local specialization below is defined so far, the
// primary template is incomplete.
template <typename DataT, int Dimensions = 1,
          access_mode AccessMode =
              (std::is_const_v<DataT> ? access_mode::read
                                      : access_mode::read_write),
          target AccessTarget = target::device,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
class accessor;

/// Provides access to work-group local memory allocated for a kernel, with
/// the same semantics and restrictions as local_accessor.
///
/// This is the SYCL 1.2.1 form of a local accessor, deprecated in SYCL 2020 in
/// favour of sycl::local_accessor.
///
/// \sa SYCL 2020 4.7.6.9.4.7. Accessor specialization with target::local.
template <typename DataT, int Dimensions, access_mode AccessMode,
          access::placeholder IsPlaceholder>
class _LIBSYCL_REIFIED_KERNEL_PARAMETER
    accessor<DataT, Dimensions, AccessMode, target::local, IsPlaceholder>
    : public detail::LocalAccessorBase<DataT, Dimensions> {
  using Base = detail::LocalAccessorBase<DataT, Dimensions>;

  // SYCL 2020 4.7.6.9.4.7: this specialization is only available for
  // access_mode::read_write and access_mode::atomic.
  static_assert(AccessMode == access_mode::read_write ||
                    AccessMode == access_mode::atomic,
                "the accessor specialization with target::local is only "
                "available for access_mode::read_write and "
                "access_mode::atomic");

  // TODO: support access_mode::atomic once the deprecated sycl::atomic class
  // is implemented.
  static_assert(AccessMode != access_mode::atomic,
                "access_mode::atomic is not supported yet for the accessor "
                "specialization with target::local");

  static_assert(!std::is_const_v<DataT>,
                "a const qualified DataT is not allowed for the accessor "
                "specialization with target::local, which is only available "
                "for access_mode::read_write");

public:
  using value_type = typename Base::value_type;
  using size_type = typename Base::size_type;

  /// Constructs an empty accessor.
  accessor() = default;

  /// Constructs an accessor instance for accessing local memory of a single
  /// DataT element within a SYCL kernel function on the queue associated with
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
  accessor([[maybe_unused]] handler &commandGroupHandlerRef,
           const property_list &propList = {})
      : Base(range<1>{1}, propList) {}

  /// Constructs an accessor instance for accessing local memory of an array of
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
  accessor(range<Dimensions> allocationSize,
           [[maybe_unused]] handler &commandGroupHandlerRef,
           const property_list &propList = {})
      : Base(allocationSize, propList) {}

  friend bool operator==(const accessor &Lhs, const accessor &Rhs) {
    return Lhs.isEqual(Rhs);
  }

  friend bool operator!=(const accessor &Lhs, const accessor &Rhs) {
    return !(Lhs == Rhs);
  }

  /// \return the size in bytes of the accessor's local memory allocation, per
  /// work-group.
  std::size_t get_size() const noexcept { return this->byte_size(); }

  /// \return the number of elements in the accessor's local memory
  /// allocation, per work-group.
  std::size_t get_count() const noexcept { return this->size(); }

private:
  friend struct std::hash<accessor>;
};

_LIBSYCL_END_NAMESPACE_SYCL

template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
          sycl::access::placeholder IsPlaceholder>
struct std::hash<sycl::accessor<DataT, Dimensions, AccessMode,
                                sycl::target::local, IsPlaceholder>> {
  std::size_t operator()(
      const sycl::accessor<DataT, Dimensions, AccessMode, sycl::target::local,
                           IsPlaceholder> &Acc) const {
    return std::hash<std::uint64_t>{}(Acc.getId());
  }
};

#endif // _LIBSYCL___IMPL_ACCESSOR_HPP
