//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the building blocks used to define SYCL properties.
///
/// A property is identified by its type alone. Rather than registering every
/// property in a central enumeration, each property states which SYCL classes
/// accept it by listing them as template arguments of its base class:
///
/// \code
///   struct in_order : detail::property_of<in_order, sycl::queue> {};
/// \endcode
///
/// From that single declaration both the compile-time trait
/// \c sycl::is_property_of and the run-time check performed by
/// \c detail::checkPropertyList are derived, so the two cannot disagree.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_PROPERTY_HELPERS_HPP
#define _LIBSYCL___IMPL_DETAIL_PROPERTY_HELPERS_HPP

#include <sycl/__impl/detail/config.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class context;
class queue;

namespace detail {

/// Kinds of SYCL classes that accept a property_list.
///
/// This enumeration is bounded by the SYCL specification: it gains an
/// enumerator only when SYCL introduces a new category of class that accepts
/// properties, never when a property or an extension is added. That is what
/// keeps property definitions themselves free of a central registry.
enum class ObjectKind : std::uint32_t {
  queue = 1u << 0,
  context = 1u << 1,
  buffer = 1u << 2,
  image = 1u << 3,
  accessor = 1u << 4,
  reduction = 1u << 5,
};

inline constexpr std::uint32_t toMask(ObjectKind Kind) {
  return static_cast<std::uint32_t>(Kind);
}

/// Mask accepted by every SYCL class, used as the identity for the fold in
/// property_list and therefore as the value of an empty property list.
inline constexpr std::uint32_t AllObjectKinds = ~std::uint32_t{0};

/// Maps a SYCL class onto its ObjectKind.
///
/// Specializing these together is deliberate: the set of SYCL classes that
/// accept properties is fixed by the specification, so unlike a registry of
/// properties this list does not grow as properties or extensions are added.
template <typename ObjT> struct ObjectKindOf;

template <> struct ObjectKindOf<queue> {
  static constexpr ObjectKind value = ObjectKind::queue;
};

template <> struct ObjectKindOf<context> {
  static constexpr ObjectKind value = ObjectKind::context;
};

/// Tags a property as applicable to \c ObjT.
///
/// A property derives from one specialization per SYCL class that accepts it;
/// sycl::is_property_of is a base class check against this tag. Mirrors the
/// detail::info_desc_tag idiom used for info descriptors.
template <typename Prop, typename ObjT> struct property_tag {};

/// Common base of every property, used by sycl::is_property.
class property_base {};

/// Identifies a property type at run time.
///
/// property_list erases the property types it was given, so it needs a
/// run-time identity for them. That identity has to be a *value* rather than
/// an address: libsycl is built with hidden visibility, so an address taken in
/// the library and the same address taken in the code using it do not match,
/// and a queue constructed with in_order would silently report
/// is_in_order() == false. An enumerator is read from this header by both
/// sides and so always agrees. DPC++ identifies properties the same way, see
/// its detail::PropKind.
///
/// Every property needs an enumerator here, which is what keeps the identity
/// free of RTTI, of exported symbols and of any dependence on linkage. The
/// cost is that a property cannot be defined outside the libsycl tree.
enum class PropKind : std::uint32_t {
  queue_in_order,
  queue_enable_profiling,
};

/// Base class of every property, naming its kind and the SYCL classes that
/// accept it.
///
/// Properties that carry data derive from this in exactly the same way as
/// properties that do not; property_list stores both through one code path.
///
/// \tparam Prop the property class itself.
/// \tparam Kind_ the property's entry in the PropKind registry.
/// \tparam ObjTs the SYCL classes that accept the property. Listing no class
///         is allowed but makes the property unusable, so it is rejected.
// Kind can live here, unlike an address-based identity, because only its value
// is ever read: a static constexpr member of a class template needs no symbol
// and no agreement between binaries.
template <typename Prop, PropKind Kind_, typename... ObjTs>
class property_of : public property_base, public property_tag<Prop, ObjTs>... {
  static_assert(sizeof...(ObjTs) > 0,
                "A property must name at least one SYCL class that accepts "
                "it, otherwise it can never be passed to anything.");

public:
  static constexpr PropKind Kind = Kind_;

  /// The ObjectKind values of ObjTs, folded into a mask.
  static constexpr std::uint32_t SupportedObjects =
      (toMask(ObjectKindOf<ObjTs>::value) | ...);
};

template <typename PropT> constexpr PropKind getPropKind() {
  return std::decay_t<PropT>::Kind;
}

/// Whether a property has to be stored to be reproduced later.
///
/// A property that holds no data carries no information beyond its own type,
/// so get_property() can rebuild it with PropT{}. Everything else has to keep
/// its value. The compile-time property extension splits on the same
/// criterion. property_list's constructor rejects a property that is empty but
/// not default constructible, which is the one case this would get wrong.
template <typename PropT>
inline constexpr bool propertyNeedsStorage = !std::is_empty_v<PropT>;

/// The properties a property_list was constructed with, type-erased.
///
/// property_list is a single concrete type, so the property types it was given
/// cannot appear in its own type and have to be erased. They are kept together
/// in one object that answers every question about them, which keeps all the
/// knowledge of the property types in the one place that still has them.
class PropertyStorageBase {
public:
  virtual ~PropertyStorageBase() = default;

  /// \return true if a property of kind Kind is held.
  virtual bool has(PropKind Kind) const noexcept = 0;

  /// \return a pointer to the held property of kind Kind, or nullptr. The
  /// kind identifies the property's type exactly, so the caller may cast the
  /// result to that type.
  virtual const void *find(PropKind Kind) const noexcept = 0;

  /// \return the SupportedObjects masks of the held properties, intersected.
  virtual std::uint32_t compatibleObjects() const noexcept = 0;
};

template <typename... PropTs>
class PropertyStorage final : public PropertyStorageBase {
public:
  explicit PropertyStorage(PropTs... Props) : MProps(std::move(Props)...) {}

  bool has(PropKind Kind) const noexcept override {
    return ((getPropKind<PropTs>() == Kind) || ...);
  }

  const void *find(PropKind Kind) const noexcept override {
    const void *Result = nullptr;
    // SYCL 2020 says nothing about a property type appearing twice in a list,
    // so the first occurrence is the one reported.
    forEachProperty([&](const auto &Prop) {
      if (!Result && getPropKind<decltype(Prop)>() == Kind)
        Result = &Prop;
    });
    return Result;
  }

  std::uint32_t compatibleObjects() const noexcept override {
    return (AllObjectKinds & ... & PropTs::SupportedObjects);
  }

private:
  template <typename FuncT> void forEachProperty(FuncT &&Func) const {
    std::apply([&](const auto &...Props) { (Func(Props), ...); }, MProps);
  }

  std::tuple<PropTs...> MProps;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_PROPERTY_HELPERS_HPP
