//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the SYCL property traits (4.5.4.1.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_PROPERTY_TRAITS_HPP
#define _LIBSYCL___IMPL_PROPERTY_TRAITS_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/property_helpers.hpp>

#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

/// Is std::true_type if propertyT is a SYCL property, std::false_type
/// otherwise.
template <typename propertyT>
struct is_property
    : std::bool_constant<std::is_base_of_v<detail::property_base, propertyT>> {
};

/// Is std::true_type if propertyT is a SYCL property that can be used with
/// syclObjectT, std::false_type otherwise.
template <typename propertyT, typename syclObjectT>
struct is_property_of
    : std::bool_constant<std::is_base_of_v<
          detail::property_tag<propertyT, syclObjectT>, propertyT>> {};

template <typename propertyT>
inline constexpr bool is_property_v = is_property<propertyT>::value;

template <typename propertyT, typename syclObjectT>
inline constexpr bool is_property_of_v =
    is_property_of<propertyT, syclObjectT>::value;

namespace detail {

template <typename PropT, typename T>
using enable_if_property_t = std::enable_if_t<is_property_v<PropT>, T>;

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_PROPERTY_TRAITS_HPP
