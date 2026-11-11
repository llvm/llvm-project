//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the common property interface that SYCL classes
/// accepting a property_list have to provide (4.5.4.1.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_PROPERTY_INTERFACE_HPP
#define _LIBSYCL___IMPL_DETAIL_PROPERTY_INTERFACE_HPP

#include <sycl/__impl/property_list.hpp>
#include <sycl/__impl/property_traits.hpp>

#include <sycl/__impl/detail/config.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// Provides has_property() and get_property() for a SYCL class.
///
/// SYCL 2020 Table 12 requires the same two member functions of every class
/// that accepts a property_list. A class mixes them in and supplies the list:
///
/// \code
///   class queue : private detail::PropertyInterface<queue> {
///   public:
///     using detail::PropertyInterface<queue>::has_property;
///     using detail::PropertyInterface<queue>::get_property;
///     ...
///   private:
///     const property_list &getPropertyList() const noexcept;
///     friend class detail::PropertyInterface<queue>;
///   };
/// \endcode
template <typename DerivedT> class PropertyInterface {
public:
  /// \return true if and only if DerivedT was constructed with a property of
  /// type PropT.
  // A property of some other SYCL class reports false rather than being
  // rejected at compile time: SYCL 2020 Table 12 puts no constraint on PropT
  // beyond it being a property.
  template <typename PropT>
  enable_if_property_t<PropT, bool> has_property() const noexcept {
    return getDerived().getPropertyList().template has_property<PropT>();
  }

  /// \return a copy of the property of type PropT that DerivedT was
  /// constructed with.
  /// \throw sycl::exception with sycl::errc::invalid if DerivedT was not
  /// constructed with a property of type PropT.
  template <typename PropT>
  enable_if_property_t<PropT, PropT> get_property() const {
    return getDerived().getPropertyList().template get_property<PropT>();
  }

private:
  const DerivedT &getDerived() const noexcept {
    return *static_cast<const DerivedT *>(this);
  }
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_PROPERTY_INTERFACE_HPP
