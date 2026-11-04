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

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class property_list;

namespace detail {
template <typename ObjT> void checkPropertyList(const property_list &PropList);

} // namespace detail

/// Collection of properties for SYCL objects. Supported properties are defined
/// by the exact object the property_list is passed to.
class property_list {
public:
  /// Constructs a property list containing the given properties.
  ///
  /// Available only when: is_property<Property>::value evaluates to true for
  /// each property in PropTs.
  ///
  /// \param props is a pack of SYCL properties.
  template <typename... PropTs,
            typename = std::enable_if_t<(is_property_v<PropTs> && ...)>>
  property_list(PropTs... props)
      : MDataLessProps(
            (std::uint64_t{0} | ... | detail::dataLessPropBit<PropTs>())),
        MCompatibleObjects(
            (detail::AllObjectKinds & ... & PropTs::SupportedObjects)) {
    static_assert(
        ((std::is_empty_v<PropTs> ? std::is_default_constructible_v<PropTs>
                                  : true) &&
         ...),
        "A property that holds no data must be default constructible, so that "
        "get_property() can reproduce it.");

    if constexpr ((detail::propertyNeedsStorage<PropTs> || ...))
      MStorage = std::make_shared<const detail::PropertyStorage<PropTs...>>(
          std::move(props)...);
  }

  property_list(const property_list &rhs) = default;
  property_list(property_list &&rhs) noexcept = default;
  property_list &operator=(const property_list &rhs) = default;
  property_list &operator=(property_list &&rhs) noexcept = default;
  ~property_list() = default;

  /// \return true if and only if the list contains a property of type PropT.
  template <typename PropT>
  detail::enable_if_property_t<PropT, bool> has_property() const noexcept {
    if constexpr (!detail::propertyNeedsStorage<PropT>)
      return (MDataLessProps & detail::dataLessPropBit<PropT>()) != 0;
    else
      return MStorage &&
             MStorage->find(detail::getPropKind<PropT>()) != nullptr;
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
    return (MCompatibleObjects & detail::toMask(Kind)) != 0;
  }

  template <typename ObjT>
  friend void detail::checkPropertyList(const property_list &);

  std::uint64_t MDataLessProps;
  std::uint32_t MCompatibleObjects;
  std::shared_ptr<const detail::PropertyStorageBase> MStorage;
};

namespace detail {

/// Verifies that every property in PropList can be applied to ObjT. Called by
/// the SYCL classes that accept a property_list.
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
