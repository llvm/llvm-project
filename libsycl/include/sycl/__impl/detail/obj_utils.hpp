//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helper functions used to navigate between SYCL interface
/// objects and their corresponding implementation objects.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_OBJ_UTILS_HPP
#define _LIBSYCL___IMPL_DETAIL_OBJ_UTILS_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// SYCL interface classes are required to contain an `impl` data member
// which points to the corresponding implementation object. The data
// member is required to be accessible by the `ImpUtils` class. SYCL
// interface classes that declare the data member private or protected
// are required to befriend the `ImpUtils` class.
struct ImplUtils {
  // Helper function to access an implementation object from a SYCL interface
  // object.
  template <typename SyclObject>
  static const decltype(SyclObject::impl) &
  getSyclObjImpl(const SyclObject &Obj) {
    assert(Obj.impl && "every constructor should create an impl");
    return Obj.impl;
  }

  // Helper function to create a SYCL interface object from an implementation.
  template <typename SyclObject, typename Impl>
  static SyclObject createSyclObjFromImpl(Impl &&ImplObj) {
    if constexpr (std::is_same_v<decltype(SyclObject::impl),
                                 std::shared_ptr<std::decay_t<Impl>>>)
      return SyclObject{ImplObj.shared_from_this()};
    else
      return SyclObject{std::forward<Impl>(ImplObj)};
  }
};

template <typename SyclObject>
auto getSyclObjImpl(const SyclObject &Obj)
    -> decltype(ImplUtils::getSyclObjImpl(Obj)) {
  return ImplUtils::getSyclObjImpl(Obj);
}

template <typename SyclObject, typename Impl>
SyclObject createSyclObjFromImpl(Impl &&ImplObj) {
  return ImplUtils::createSyclObjFromImpl<SyclObject>(
      std::forward<Impl>(ImplObj));
}

// SYCL 2020 4.5.2. Common reference semantics (std::hash support).
template <typename T> struct HashBase {
  size_t operator()(const T &Obj) const {
    auto &Impl = sycl::detail::getSyclObjImpl(Obj);
    return std::hash<std::decay_t<decltype(Impl)>>{}(Impl);
  }
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_OBJ_UTILS_HPP
