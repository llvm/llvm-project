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
/// A property states which SYCL classes accept it by listing them as template
/// arguments of its base class:
///
/// \code
///   struct in_order : detail::property_of<in_order,
///                                         detail::PropKind::QueueInOrder,
///                                         sycl::queue> {};
/// \endcode
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_PROPERTY_HELPERS_HPP
#define _LIBSYCL___IMPL_DETAIL_PROPERTY_HELPERS_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class context;
class queue;

namespace detail {

/// Kinds of SYCL classes that accept a property_list. An enumerator is added
/// here when such a class is implemented.
enum class ObjectKind : std::uint32_t {
  Queue = 1u << 0,
  Context = 1u << 1,
};

inline constexpr std::uint32_t toMask(ObjectKind Kind) {
  return static_cast<std::uint32_t>(Kind);
}

/// Mask accepted by every SYCL class, and therefore the value of an empty
/// property list.
inline constexpr std::uint32_t AllObjectKinds = ~std::uint32_t{0};

/// Maps a SYCL class onto its ObjectKind.
template <typename ObjT> struct ObjectKindOf;

template <> struct ObjectKindOf<queue> {
  static constexpr ObjectKind value = ObjectKind::Queue;
};

template <> struct ObjectKindOf<context> {
  static constexpr ObjectKind value = ObjectKind::Context;
};

/// Tags a property as applicable to \c ObjT, used by sycl::is_property_of.
template <typename Prop, typename ObjT> struct property_tag {};

/// Common base of every property, used by sycl::is_property.
class property_base {};

/// Identifies a property type at run time, as property_list erases the
/// property types it was given. Every property needs an enumerator here.
enum class PropKind : std::uint32_t {
  QueueInOrder,
  QueueEnableProfiling,
  // Must be the last enumerator.
  Count,
};

// property_list keeps the kinds of the properties without data it holds in a
// 64 bit mask.
static_assert(static_cast<std::uint32_t>(PropKind::Count) <= 64,
              "PropKind has outgrown the mask held by property_list.");

/// Base class of every property, naming its kind and the SYCL classes that
/// accept it.
///
/// \tparam Prop the property class itself.
/// \tparam KindValue the property's entry in the PropKind registry.
/// \tparam ObjTs the SYCL classes that accept the property.
template <typename Prop, PropKind KindValue, typename... ObjTs>
class property_of : public property_base, public property_tag<Prop, ObjTs>... {
  static_assert(sizeof...(ObjTs) > 0,
                "A property must name at least one SYCL class that accepts "
                "it.");

public:
  static constexpr PropKind Kind = KindValue;

  static constexpr std::uint32_t SupportedObjects =
      (toMask(ObjectKindOf<ObjTs>::value) | ...);
};

template <typename PropT> constexpr PropKind getPropKind() {
  return std::decay_t<PropT>::Kind;
}

/// Whether a property has to be stored to be reproduced later. A property
/// that holds no data carries no information beyond its own type, so
/// get_property() can rebuild it with PropT{}.
template <typename PropT>
inline constexpr bool propertyNeedsStorage = !std::is_empty_v<PropT>;

/// Bit identifying a property without data in the mask held by property_list.
/// Zero for a property that holds data, which PropertyStorage keeps instead.
template <typename PropT> constexpr std::uint64_t dataLessPropBit() {
  if constexpr (propertyNeedsStorage<PropT>) {
    return std::uint64_t{0};
  } else {
    constexpr auto Index = static_cast<std::uint32_t>(getPropKind<PropT>());
    static_assert(Index < 64, "The property kind does not fit the mask held "
                              "by property_list.");
    return std::uint64_t{1} << Index;
  }
}

/// The properties a property_list was constructed with, type-erased.
///
/// property_list is a single concrete type, so the property types it was given
/// cannot appear in its own type and have to be erased.
class PropertyStorageBase {
public:
  virtual ~PropertyStorageBase() = default;

  /// \return a pointer to the held property of kind Kind, or nullptr. The
  /// kind identifies the property's type exactly, so the caller may cast the
  /// result to that type.
  virtual const void *find(PropKind Kind) const noexcept = 0;
};

template <typename... PropTs>
class PropertyStorage final : public PropertyStorageBase {
public:
  explicit PropertyStorage(PropTs... Props) : MProps(std::move(Props)...) {}

  const void *find(PropKind Kind) const noexcept override {
    const void *Result = nullptr;
    // The fold stops at the first property of this kind: SYCL 2020 says
    // nothing about a property type appearing twice in a list.
    std::apply(
        [&](const auto &...Props) {
          return ((getPropKind<decltype(Props)>() == Kind &&
                   (Result = &Props) != nullptr) ||
                  ...);
        },
        MProps);
    return Result;
  }

private:
  std::tuple<PropTs...> MProps;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_PROPERTY_HELPERS_HPP
