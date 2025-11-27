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

#ifndef _LIBSYCL___IMPL_DETAIL_OBJ_BASE_HPP
#define _LIBSYCL___IMPL_DETAIL_OBJ_BASE_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cassert>
#include <optional>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

template <typename Impl, typename SyclObject> class ObjBase;
template <typename Impl, typename SyclObject>
class ObjBase<Impl *, SyclObject> {
public:
  using ImplType = Impl;
  using ImplPtrType = Impl *;
  using Base = ObjBase<Impl *, SyclObject>;

protected:
  ImplPtrType impl;

  explicit ObjBase(ImplPtrType pImpl) : impl(pImpl) {}
  ObjBase() = default;

  static SyclObject createSyclProxy(ImplPtrType impl) {
    return SyclObject(impl);
  }

  ImplType &getImpl() const { return *impl; }

  template <class Obj>
  friend const typename Obj::ImplType &getSyclObjImpl(const Obj &Object);

  template <class Obj>
  friend Obj createSyclObjFromImpl(
      std::add_lvalue_reference_t<const typename Obj::ImplPtrType> ImplObj);
};

template <class Obj>
const typename Obj::ImplType &getSyclObjImpl(const Obj &Object) {
  return *Object.impl;
}

template <class Obj>
Obj createSyclObjFromImpl(
    std::add_lvalue_reference_t<const typename Obj::ImplPtrType> ImplObj) {
  return Obj::Base::createSyclProxy(ImplObj);
}

// std::hash support (4.5.2. Common reference semantics)
template <typename T> struct HashBase {
  size_t operator()(const T &Obj) const {
#ifdef __SYCL_DEVICE_ONLY__
    (void)Obj;
    return 0;
#else
    auto &Impl = sycl::detail::getSyclObjImpl(Obj);
    return std::hash<std::decay_t<decltype(Impl)>>{}(Impl);
#endif
  }
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_OBJ_BASE_HPP
