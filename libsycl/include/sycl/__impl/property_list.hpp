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

_LIBSYCL_BEGIN_NAMESPACE_SYCL

/// Collection of properties for SYCL objects. Supported properties are defined
/// by exact object the property_list passed to.
// Just a placeholder for initial stage.
class property_list {
public:
  template <typename... Properties>
  property_list([[maybe_unused]] Properties... props) {}
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_PROPERTY_LIST_HPP
