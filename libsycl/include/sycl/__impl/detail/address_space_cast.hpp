//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helpers to cast a pointer between SYCL address spaces.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_ADDRESS_SPACE_CAST_HPP
#define _LIBSYCL___IMPL_DETAIL_ADDRESS_SPACE_CAST_HPP

#include <sycl/__impl/access_enums.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/decorated_type.hpp>

#include <sycl/__spirv/types.hpp>

#include <__clang_spirv_builtins.h>

#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// \return true if a pointer in address space Src can be cast to address
/// space Dst.
inline constexpr bool isAddressSpaceCastPossible(access::address_space Src,
                                                 access::address_space Dst) {
  // The constant address space is not interchangeable with any other one.
  if (Src == access::address_space::constant_space ||
      Dst == access::address_space::constant_space)
    return Src == Dst;

  return Src == Dst || Src == access::address_space::generic_space ||
         Dst == access::address_space::generic_space;
}

/// Casts Ptr to address space Space without a run-time check.
// It is used whenever the spec allows that (e.g. if it's UB if runtime
// pointers doesn't point to a proper allocation).
template <access::address_space Space, typename ElementType>
auto staticAddressCast(ElementType *Ptr) {
  using Pointee =
      std::remove_pointer_t<typename RemoveDecoration<ElementType *>::type>;
  using DstT = DecoratedType_t<Pointee, Space> *;

  static_assert(isAddressSpaceCastPossible(deduceAS<ElementType *>, Space),
                "the source and the destination address space of a static "
                "address space cast must be compatible");

  // Note: for extended address spaces (extensions) on SPIRV and all casts on
  // HIP/CUDA reinterpret_cast can be not enough and can lead to warnings, use
  // C-style cast instead in advance.
  return (DstT)Ptr;
}

/// Casts Ptr to address space Space, checking at run time that Ptr really
/// points into that address space. \return nullptr if it does not.
template <access::address_space Space, typename ElementType>
auto dynamicAddressCast(ElementType *Ptr) {
  using Pointee =
      std::remove_pointer_t<typename RemoveDecoration<ElementType *>::type>;
  using DstT = DecoratedType_t<Pointee, Space> *;

  constexpr access::address_space SrcSpace = deduceAS<ElementType *>;

  if constexpr (!isAddressSpaceCastPossible(SrcSpace, Space)) {
    // SYCL 2020 4.7.7.1: an implementation must return nullptr if the value of
    // the pointer is not compatible with the destination address space.
    return (DstT) nullptr;
  } else if constexpr (SrcSpace != access::address_space::generic_space ||
                       Space == access::address_space::generic_space) {
    // No run-time check is needed: the source already is in the destination
    // address space, or the destination is the generic address space.
    return staticAddressCast<Space>(Ptr);
  } else {
#ifdef __SYCL_DEVICE_ONLY__
    using RemoveCvT = std::remove_cv_t<Pointee>;
    auto *Unqualified = const_cast<RemoveCvT *>(Ptr);
    // SPIRV OpGenericCastToPtrExplicit requirement:
    // Storage must be one of the following literal values from Storage Class:
    // Workgroup, CrossWorkgroup, or Function.
    if constexpr (Space == access::address_space::global_space)
      return (DstT)__spirv_GenericCastToPtrExplicit_ToGlobal(
          Unqualified, static_cast<int>(__spirv::StorageClass::CrossWorkgroup));
    else if constexpr (Space == access::address_space::local_space)
      return (DstT)__spirv_GenericCastToPtrExplicit_ToLocal(
          Unqualified, static_cast<int>(__spirv::StorageClass::Workgroup));
    else if constexpr (Space == access::address_space::private_space)
      return (DstT)__spirv_GenericCastToPtrExplicit_ToPrivate(
          Unqualified, static_cast<int>(__spirv::StorageClass::Function));
    else
      // The constant address space, which has no storage class accepted by
      // the builtin, cannot reach this branch: a cast to it from the generic
      // address space is rejected by isAddressSpaceCastPossible, and a cast
      // from the constant address space itself takes the branch above.
      static_assert(Space != access::address_space::constant_space,
                    "unexpected destination address space");
#else
    // There is a single address space on host, so the cast always succeeds.
    return staticAddressCast<Space>(Ptr);
#endif
  }
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_ADDRESS_SPACE_CAST_HPP
