//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helper functions for tranformation between implementation
/// and SYCL's interface objects.
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

// Note! This class relies on the fact that all SYCL interface
// classes contain "impl" field that points to implementation object. "impl"
// field should be accessible from this class.
struct ImplUtils {
  // Helper function for extracting implementation from SYCL's interface
  // objects.
  template <class Obj>
  static const decltype(Obj::impl) &getSyclObjImpl(const Obj &SyclObj) {
    assert(SyclObj.impl && "every constructor should create an impl");
    return SyclObj.impl;
  }

  // Helper function for creation SYCL interface objects from implementations.
  template <typename SyclObject, typename Impl>
  static SyclObject createSyclObjFromImpl(Impl &&ImplObj) {
    if constexpr (std::is_same_v<decltype(SyclObject::impl),
                                 std::shared_ptr<std::decay_t<Impl>>>)
      return SyclObject{ImplObj.shared_from_this()};
    else
      return SyclObject{std::forward<Impl>(ImplObj)};
  }
};

template <class Obj>
auto getSyclObjImpl(const Obj &SyclObj)
    -> decltype(ImplUtils::getSyclObjImpl(SyclObj)) {
  return ImplUtils::getSyclObjImpl(SyclObj);
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
