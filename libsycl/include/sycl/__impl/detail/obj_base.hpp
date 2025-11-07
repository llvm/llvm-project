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
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

template <class Impl, class SyclObject> class ObjBase {
public:
  using ImplType = Impl;
  using Base = ObjBase<Impl, SyclObject>;

protected:
  ImplType &impl;

  explicit ObjBase(ImplType &pImpl) : impl(pImpl) {}
  ObjBase() = default;

  static SyclObject createSyclProxy(ImplType &impl) { return SyclObject(impl); }

  template <class Obj>
  friend const typename Obj::ImplType &getSyclObjImpl(const Obj &Object);

  template <class Obj>
  friend Obj createSyclObjFromImpl(
      std::add_lvalue_reference_t<typename Obj::ImplType> ImplObj);
};

template <class Obj>
const typename Obj::ImplType &getSyclObjImpl(const Obj &Object) {
  return Object.impl;
}

template <class Obj>
Obj createSyclObjFromImpl(
    std::add_lvalue_reference_t<typename Obj::ImplType> ImplObj) {
  return Obj::Base::createSyclProxy(ImplObj);
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_OBJ_BASE_HPP
