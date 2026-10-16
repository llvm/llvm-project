//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL property_list type, which
/// contains zero or more properties and is used as an optional parameter in
/// SYCL runtime classes constructors. Each of those properties augments the
/// semantics of the class with a particular feature.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_PROPERTY_LIST_HPP
#define _LIBSYCL___IMPL_PROPERTY_LIST_HPP

#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/property_traits.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/property_helpers.hpp>

#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class property_list;

namespace detail {
template <typename ObjT> void checkPropertyList(const property_list &PropList);

} // namespace detail

/// Collection of properties for SYCL objects. Supported properties are defined
/// by the exact object the property_list is passed to.
///
/// A list is a handle on one type-erased detail::PropertyStorage holding the
/// properties it was constructed with, shared with its copies, and null when
/// the list is empty.
class property_list {
public:
  /// Constructs a property list containing the given properties.
  ///
  /// Available only when: is_property<Property>::value evaluates to true for
  /// each property in PropTs.
  ///
  /// \param props is a pack of SYCL properties.
  // SYCL 2020 Table 13 phrases this as "available only when", so the
  // constraint has to remove the constructor from the overload set rather
  // than fire a static_assert: std::is_constructible_v<property_list, int> is
  // required to be false.
  // Copy and move constructors are declared below and, being non-templates,
  // are preferred over this constructor when copying a property_list.
  template <typename... PropTs,
            typename = std::enable_if_t<(is_property_v<PropTs> && ...)>>
  property_list(PropTs... props) {
    static_assert(
        ((std::is_empty_v<PropTs> ? std::is_default_constructible_v<PropTs>
                                  : true) &&
         ...),
        "A property that holds no data must be default constructible, so that "
        "get_property() can reproduce it.");

    if constexpr (sizeof...(PropTs) > 0)
      MStorage = std::make_shared<const detail::PropertyStorage<PropTs...>>(
          std::move(props)...);
  }

  property_list(const property_list &rhs) = default;
  property_list(property_list &&rhs) = default;
  property_list &operator=(const property_list &rhs) = default;
  property_list &operator=(property_list &&rhs) = default;
  ~property_list() = default;

  /// \return true if and only if the list contains a property of type PropT.
  template <typename PropT>
  detail::enable_if_property_t<PropT, bool> has_property() const noexcept {
    return MStorage && MStorage->has(detail::getPropKind<PropT>());
  }

  /// \return a copy of the property of type PropT held by this list.
  /// \throw sycl::exception with sycl::errc::invalid if the list does not
  /// contain a property of type PropT.
  template <typename PropT>
  detail::enable_if_property_t<PropT, PropT> get_property() const {
    if constexpr (!detail::propertyNeedsStorage<PropT>) {
      if (!has_property<PropT>())
        throw exception(make_error_code(errc::invalid),
                        "The property is not found");
      // Nothing was stored, because the type alone says all there is to know.
      return PropT{};
    } else {
      const void *Value =
          MStorage ? MStorage->find(detail::getPropKind<PropT>()) : nullptr;
      if (!Value)
        throw exception(make_error_code(errc::invalid),
                        "The property is not found");
      // The kind matched, so PropT is the type that was stored.
      return *static_cast<const PropT *>(Value);
    }
  }

private:
  /// \return true if every property in this list can be applied to Kind.
  bool isCompatibleWith(detail::ObjectKind Kind) const noexcept {
    // An empty list can be passed to anything.
    const std::uint32_t Mask =
        MStorage ? MStorage->compatibleObjects() : detail::AllObjectKinds;
    return (Mask & detail::toMask(Kind)) != 0;
  }

  template <typename ObjT>
  friend void detail::checkPropertyList(const property_list &);

  // Null when the list is empty. Shared rather than unique so that copying a
  // list, which SYCL requires to carry the same properties, need not clone
  // them.
  std::shared_ptr<const detail::PropertyStorageBase> MStorage;
};

namespace detail {

/// Verifies that every property in PropList can be applied to ObjT.
///
/// Called by the SYCL classes that accept a property_list. The whole list is
/// validated with a single mask test, because property_list has already
/// intersected the supported-class masks of its properties.
///
/// \throw sycl::exception with sycl::errc::invalid if PropList holds a
/// property that ObjT does not accept.
template <typename ObjT> void checkPropertyList(const property_list &PropList) {
  if (!PropList.isCompatibleWith(ObjectKindOf<ObjT>::value))
    throw exception(make_error_code(errc::invalid),
                    "The property list contains a property that is not "
                    "supported by this SYCL object");
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_PROPERTY_LIST_HPP
