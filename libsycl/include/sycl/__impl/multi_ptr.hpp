//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 multi_ptr class
/// (4.7.7.1.) and of the explicit pointer aliases (4.7.7.2.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_MULTI_PTR_HPP
#define _LIBSYCL___IMPL_MULTI_PTR_HPP

#include <sycl/__impl/access_enums.hpp>
#include <sycl/__impl/detail/address_space_cast.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/decorated_type.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

template <typename T> struct remove_decoration {
  using type = typename detail::RemoveDecoration<T>::type;
};

template <typename T>
using remove_decoration_t = typename remove_decoration<T>::type;

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress = access::decorated::legacy>
class multi_ptr;

namespace detail {

// Relational operators are the same for every multi_ptr specialization. The
// macro takes the name of the decorated pointer type alias of the enclosing
// class.
#define _LIBSYCL_MULTI_PTR_REL_OPS(PtrT)                                       \
  _LIBSYCL_MULTI_PTR_REL_OP(PtrT, ==)                                          \
  _LIBSYCL_MULTI_PTR_REL_OP(PtrT, !=)                                          \
  _LIBSYCL_MULTI_PTR_REL_OP(PtrT, <)                                           \
  _LIBSYCL_MULTI_PTR_REL_OP(PtrT, >)                                           \
  _LIBSYCL_MULTI_PTR_REL_OP(PtrT, <=)                                          \
  _LIBSYCL_MULTI_PTR_REL_OP(PtrT, >=)

#define _LIBSYCL_MULTI_PTR_REL_OP(PtrT, OP)                                    \
  friend bool operator OP(const multi_ptr &Lhs, const multi_ptr &Rhs) {        \
    return Lhs.MPtr OP Rhs.MPtr;                                               \
  }                                                                            \
  friend bool operator OP(const multi_ptr &Lhs, std::nullptr_t) {              \
    return Lhs.MPtr OP static_cast<PtrT>(nullptr);                             \
  }                                                                            \
  friend bool operator OP(std::nullptr_t, const multi_ptr &Rhs) {              \
    return static_cast<PtrT>(nullptr) OP Rhs.MPtr;                             \
  }

} // namespace detail

/// Pointer that carries the address space it points into.
///
/// \tparam ElementType the type of the pointed-to element.
/// \tparam Space the address space the pointer points into.
/// \tparam DecorateAddress whether the interface exposes the address space
/// qualified pointer type.
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr {
  using DecoratedPtrT = detail::DecoratedType_t<ElementType, Space> *;

public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = ElementType;
  using pointer = std::conditional_t<is_decorated, DecoratedPtrT,
                                     std::add_pointer_t<value_type>>;
  using reference =
      std::conditional_t<is_decorated,
                         detail::DecoratedType_t<ElementType, Space> &,
                         std::add_lvalue_reference_t<value_type>>;
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;

  // The decorated and the undecorated type may only differ in the address
  // space. A multi_ptr in the constant address space therefore requires a
  // const value_type, because that address space is read-only.
  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                               std::add_pointer_t<value_type>>,
                "the pointer type must decay to a pointer to value_type");
  static_assert(std::is_same_v<remove_decoration_t<reference>,
                               std::add_lvalue_reference_t<value_type>>,
                "the reference type must decay to a reference to value_type");
  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  ~multi_ptr() = default;

  explicit multi_ptr(typename multi_ptr<ElementType, Space,
                                        access::decorated::yes>::pointer Ptr)
      : MPtr(Ptr) {}

  multi_ptr(std::nullptr_t) : MPtr(nullptr) {}

  // TODO: add the constructors from accessor and local_accessor, and the
  // corresponding deduction guides, once accessors are supported.

  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

  multi_ptr &operator=(std::nullptr_t) {
    MPtr = nullptr;
    return *this;
  }

  template <access::address_space AS, access::decorated IsDecorated,
            std::enable_if_t<Space == access::address_space::generic_space &&
                                 AS != access::address_space::constant_space,
                             bool> = true>
  multi_ptr &operator=(const multi_ptr<value_type, AS, IsDecorated> &Other) {
    MPtr = detail::staticAddressCast<Space>(Other.get_decorated());
    return *this;
  }

  template <access::address_space AS, access::decorated IsDecorated,
            std::enable_if_t<Space == access::address_space::generic_space &&
                                 AS != access::address_space::constant_space,
                             bool> = true>
  multi_ptr &operator=(multi_ptr<value_type, AS, IsDecorated> &&Other) {
    MPtr = detail::staticAddressCast<Space>(Other.get_decorated());
    return *this;
  }

  reference operator[](std::ptrdiff_t Index) const { return MPtr[Index]; }

  reference operator*() const { return *MPtr; }

  pointer operator->() const { return get(); }

  pointer get() const { return MPtr; }

  /// \return the pointer without the address space decoration.
  std::add_pointer_t<value_type> get_raw() const { return MPtr; }

  DecoratedPtrT get_decorated() const { return MPtr; }

  __SYCL2020_DEPRECATED("use get() instead")
  operator pointer() const { return get(); }

  // Conversions to a narrower address space. SYCL 2020 4.7.7.1 leaves the
  // result undefined if the pointer does not address the destination address
  // space, so no run-time check is performed. sycl::address_space_cast is the
  // checked alternative.
  template <access::address_space OtherSpace,
            access::decorated OtherIsDecorated,
            access::address_space RelaySpace = Space,
            std::enable_if_t<
                RelaySpace == access::address_space::generic_space &&
                    (OtherSpace == access::address_space::private_space ||
                     OtherSpace == access::address_space::global_space ||
                     OtherSpace == access::address_space::local_space),
                bool> = true>
  explicit
  operator multi_ptr<value_type, OtherSpace, OtherIsDecorated>() const {
    return multi_ptr<value_type, OtherSpace, OtherIsDecorated>{
        detail::staticAddressCast<OtherSpace>(MPtr)};
  }

  // The same conversion to a multi_ptr of const data. Only available when
  // value_type is not already const, otherwise it would have the same
  // signature as the conversion above.
  template <
      access::address_space OtherSpace, access::decorated OtherIsDecorated,
      access::address_space RelaySpace = Space, typename RelayT = value_type,
      std::enable_if_t<RelaySpace == access::address_space::generic_space &&
                           !std::is_const_v<RelayT> &&
                           (OtherSpace ==
                                access::address_space::private_space ||
                            OtherSpace == access::address_space::global_space ||
                            OtherSpace == access::address_space::local_space),
                       bool> = true>
  explicit
  operator multi_ptr<const value_type, OtherSpace, OtherIsDecorated>() const {
    return multi_ptr<const value_type, OtherSpace, OtherIsDecorated>{
        detail::staticAddressCast<OtherSpace>(MPtr)};
  }

  // TODO: check if cast is enough or I should use getDecorated();
  /// Implicit conversion to a multi_ptr<void>.
  template <access::decorated IsDecorated, typename RelayT = value_type,
            std::enable_if_t<!std::is_const_v<RelayT>, bool> = true>
  operator multi_ptr<void, Space, IsDecorated>() const {
    using DstT = detail::DecoratedType_t<void, Space> *;
    return multi_ptr<void, Space, IsDecorated>{static_cast<DstT>(MPtr)};
  }

  /// Implicit conversion to a multi_ptr<const void>.
  template <access::decorated IsDecorated, typename RelayT = value_type,
            std::enable_if_t<std::is_const_v<RelayT>, bool> = true>
  operator multi_ptr<const void, Space, IsDecorated>() const {
    using DstT = detail::DecoratedType_t<const void, Space> *;
    return multi_ptr<const void, Space, IsDecorated>{static_cast<DstT>(MPtr)};
  }

  /// Implicit conversion to a multi_ptr of const data.
  template <access::decorated IsDecorated>
  operator multi_ptr<const value_type, Space, IsDecorated>() const {
    return multi_ptr<const value_type, Space, IsDecorated>{MPtr};
  }

  /// Implicit conversion to the non-decorated version of multi_ptr.
  template <bool Dependent = is_decorated,
            std::enable_if_t<Dependent, bool> = true>
  operator multi_ptr<value_type, Space, access::decorated::no>() const {
    return multi_ptr<value_type, Space, access::decorated::no>{MPtr};
  }

  /// Implicit conversion to the decorated version of multi_ptr.
  template <bool Dependent = is_decorated,
            std::enable_if_t<!Dependent, bool> = true>
  operator multi_ptr<value_type, Space, access::decorated::yes>() const {
    return multi_ptr<value_type, Space, access::decorated::yes>{MPtr};
  }

  /// Prefetches a number of elements into the global memory cache.
  ///
  /// SYCL 2020 4.7.7.1: this is an implementation-defined optimization that
  /// does not affect the functional behavior of a kernel, so doing nothing is
  /// a conforming implementation.
  template <
      access::address_space AS = Space,
      std::enable_if_t<AS == access::address_space::global_space, bool> = true>
  void prefetch(std::size_t) const {
    // TODO: lower to a prefetch once a clang builtin usable from SYCL device
    // code is available (__spirv_ocl_prefetch).
  }

  friend multi_ptr &operator++(multi_ptr &Mp) {
    ++Mp.MPtr;
    return Mp;
  }
  friend multi_ptr operator++(multi_ptr &Mp, int) {
    multi_ptr Tmp{Mp};
    ++Mp.MPtr;
    return Tmp;
  }
  friend multi_ptr &operator--(multi_ptr &Mp) {
    --Mp.MPtr;
    return Mp;
  }
  friend multi_ptr operator--(multi_ptr &Mp, int) {
    multi_ptr Tmp{Mp};
    --Mp.MPtr;
    return Tmp;
  }
  friend multi_ptr &operator+=(multi_ptr &Lhs, difference_type R) {
    Lhs.MPtr += R;
    return Lhs;
  }
  friend multi_ptr &operator-=(multi_ptr &Lhs, difference_type R) {
    Lhs.MPtr -= R;
    return Lhs;
  }
  friend multi_ptr operator+(const multi_ptr &Lhs, difference_type R) {
    return multi_ptr{Lhs.MPtr + R};
  }
  friend multi_ptr operator-(const multi_ptr &Lhs, difference_type R) {
    return multi_ptr{Lhs.MPtr - R};
  }
  _LIBSYCL_MULTI_PTR_REL_OPS(DecoratedPtrT)

private:
  DecoratedPtrT MPtr = nullptr;

  template <typename, access::address_space, access::decorated>
  friend class multi_ptr;
};

/// Specialization of multi_ptr for void.
template <access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<void, Space, DecorateAddress> {
  using DecoratedPtrT = detail::DecoratedType_t<void, Space> *;

public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = void;
  using pointer = std::conditional_t<is_decorated, DecoratedPtrT, void *>;
  using difference_type = std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>, void *>);
  static_assert(DecorateAddress != access::decorated::legacy);

  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  ~multi_ptr() = default;

  explicit multi_ptr(
      typename multi_ptr<void, Space, access::decorated::yes>::pointer Ptr)
      : MPtr(Ptr) {}

  multi_ptr(std::nullptr_t) : MPtr(nullptr) {}

  // TODO: add the constructors from accessor and local_accessor once
  // accessors are supported.

  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

  multi_ptr &operator=(std::nullptr_t) {
    MPtr = nullptr;
    return *this;
  }

  pointer get() const { return MPtr; }

  operator pointer() const { return get(); }

  /// Explicit conversion to a multi_ptr of a concrete element type.
  template <typename ElementType>
  explicit operator multi_ptr<ElementType, Space, DecorateAddress>() const {
    using DstT = detail::DecoratedType_t<ElementType, Space> *;
    return multi_ptr<ElementType, Space, DecorateAddress>{
        reinterpret_cast<DstT>(MPtr)};
  }

  /// Implicit conversion to the non-decorated version of multi_ptr.
  template <bool Dependent = is_decorated,
            std::enable_if_t<Dependent, bool> = true>
  operator multi_ptr<void, Space, access::decorated::no>() const {
    return multi_ptr<void, Space, access::decorated::no>{MPtr};
  }

  /// Implicit conversion to the decorated version of multi_ptr.
  template <bool Dependent = is_decorated,
            std::enable_if_t<!Dependent, bool> = true>
  operator multi_ptr<void, Space, access::decorated::yes>() const {
    return multi_ptr<void, Space, access::decorated::yes>{MPtr};
  }

  /// Implicit conversion to a multi_ptr<const void>.
  operator multi_ptr<const void, Space, DecorateAddress>() const {
    using DstT = detail::DecoratedType_t<const void, Space> *;
    return multi_ptr<const void, Space, DecorateAddress>{
        reinterpret_cast<DstT>(MPtr)};
  }

  _LIBSYCL_MULTI_PTR_REL_OPS(DecoratedPtrT)

private:
  DecoratedPtrT MPtr = nullptr;

  template <typename, access::address_space, access::decorated>
  friend class multi_ptr;
};

/// Specialization of multi_ptr for const void.
template <access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<const void, Space, DecorateAddress> {
  using DecoratedPtrT = detail::DecoratedType_t<const void, Space> *;

public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = const void;
  using pointer = std::conditional_t<is_decorated, DecoratedPtrT, const void *>;
  using difference_type = std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>, const void *>);
  static_assert(DecorateAddress != access::decorated::legacy);

  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  ~multi_ptr() = default;

  explicit multi_ptr(typename multi_ptr<const void, Space,
                                        access::decorated::yes>::pointer Ptr)
      : MPtr(Ptr) {}
  multi_ptr(std::nullptr_t) : MPtr(nullptr) {}

  // TODO: add the constructors from accessor and local_accessor once
  // accessors are supported.

  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

  multi_ptr &operator=(std::nullptr_t) {
    MPtr = nullptr;
    return *this;
  }

  pointer get() const { return MPtr; }

  operator pointer() const { return get(); }

  /// Explicit conversion to a multi_ptr of a concrete element type. Only
  /// available for a const element type, const void cannot be dropped.
  template <typename ElementType,
            std::enable_if_t<std::is_const_v<ElementType>, bool> = true>
  explicit operator multi_ptr<ElementType, Space, DecorateAddress>() const {
    using DstT = detail::DecoratedType_t<ElementType, Space> *;
    return multi_ptr<ElementType, Space, DecorateAddress>{
        reinterpret_cast<DstT>(MPtr)};
  }

  /// Implicit conversion to the non-decorated version of multi_ptr.
  template <bool Dependent = is_decorated,
            std::enable_if_t<Dependent, bool> = true>
  operator multi_ptr<const void, Space, access::decorated::no>() const {
    return multi_ptr<const void, Space, access::decorated::no>{MPtr};
  }

  /// Implicit conversion to the decorated version of multi_ptr.
  template <bool Dependent = is_decorated,
            std::enable_if_t<!Dependent, bool> = true>
  operator multi_ptr<const void, Space, access::decorated::yes>() const {
    return multi_ptr<const void, Space, access::decorated::yes>{MPtr};
  }

  _LIBSYCL_MULTI_PTR_REL_OPS(DecoratedPtrT)

private:
  DecoratedPtrT MPtr = nullptr;

  template <typename, access::address_space, access::decorated>
  friend class multi_ptr;
};

/// Deprecated SYCL 1.2.1 interface of multi_ptr, kept because
/// access::decorated::legacy is the default of the DecorateAddress template
/// parameter and the explicit pointer aliases default to it.
template <typename ElementType, access::address_space Space>
class [[deprecated]] multi_ptr<ElementType, Space, access::decorated::legacy> {
public:
  using value_type = ElementType;
  using element_type = ElementType;
  using difference_type = std::ptrdiff_t;

  using pointer_t =
      multi_ptr<ElementType, Space, access::decorated::yes>::pointer;
  using const_pointer_t =
      multi_ptr<const ElementType, Space, access::decorated::yes>::pointer;
  using reference_t =
      multi_ptr<ElementType, Space, access::decorated::yes>::reference;
  using const_reference_t =
      multi_ptr<const ElementType, Space, access::decorated::yes>::reference;

  static constexpr access::address_space address_space = Space;

  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  ~multi_ptr() = default;

#ifdef __SYCL_DEVICE_ONLY__
  // In a host compilation the decorated pointer type is the same type as
  // ElementType *, so this constructor would be a duplicate of the one
  // below.
  multi_ptr(pointer_t Ptr) : MPtr(Ptr) {}
#endif

  multi_ptr(ElementType *Ptr) : MPtr(detail::dynamicAddressCast<Space>(Ptr)) {}

  multi_ptr(std::nullptr_t) : MPtr(nullptr) {}

  // TODO: add the constructors from accessor and local_accessor once
  // accessors are supported.

  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr &operator=(pointer_t Ptr) {
    MPtr = Ptr;
    return *this;
  }
#endif

  multi_ptr &operator=(ElementType *Ptr) {
    MPtr = detail::dynamicAddressCast<Space>(Ptr);
    return *this;
  }

  multi_ptr &operator=(std::nullptr_t) {
    MPtr = nullptr;
    return *this;
  }

  // The access operators of the deprecated interface yield the undecorated
  // type. pointer_t and reference_t, which carry the address space, are only
  // exposed by get() and get_decorated() below.
  friend ElementType &operator*(const multi_ptr &Mp) { return *Mp.MPtr; }
  ElementType *operator->() const { return MPtr; }

  pointer_t get() const { return MPtr; }
  pointer_t get_decorated() const { return MPtr; }
  std::add_pointer_t<element_type> get_raw() const { return MPtr; }

  /// Implicit conversion to the underlying pointer type.
  operator ElementType *() const { return MPtr; }

  /// Implicit conversion to a multi_ptr of const data.
  template <typename RelayElementType = ElementType,
            std::enable_if_t<!std::is_const_v<RelayElementType>, bool> = true>
  operator multi_ptr<const ElementType, Space, access::decorated::legacy>()
      const {
    using DstT = detail::DecoratedType_t<const ElementType, Space> *;
    return multi_ptr<const ElementType, Space, access::decorated::legacy>{
        static_cast<DstT>(MPtr)};
  }

  /// Implicit conversion to a multi_ptr<void>.
  template <typename RelayElementType = ElementType,
            std::enable_if_t<!std::is_const_v<RelayElementType>, bool> = true>
  operator multi_ptr<void, Space, access::decorated::legacy>() const {
    using DstT = detail::DecoratedType_t<void, Space> *;
    return multi_ptr<void, Space, access::decorated::legacy>{
        reinterpret_cast<DstT>(MPtr)};
  }

  /// Implicit conversion to a multi_ptr<const void>.
  template <typename RelayElementType = ElementType,
            std::enable_if_t<std::is_const_v<RelayElementType>, bool> = true>
  operator multi_ptr<const void, Space, access::decorated::legacy>() const {
    using DstT = detail::DecoratedType_t<const void, Space> *;
    return multi_ptr<const void, Space, access::decorated::legacy>{
        reinterpret_cast<DstT>(MPtr)};
  }

  // See multi_ptr::prefetch of the non-legacy interface.
  template <
      access::address_space AS = Space,
      std::enable_if_t<AS == access::address_space::global_space, bool> = true>
  void prefetch(std::size_t NumElements) const {
    (void)NumElements;
  }

  friend multi_ptr &operator++(multi_ptr &Mp) {
    ++Mp.MPtr;
    return Mp;
  }
  friend multi_ptr operator++(multi_ptr &Mp, int) {
    multi_ptr Tmp{Mp};
    ++Mp.MPtr;
    return Tmp;
  }
  friend multi_ptr &operator--(multi_ptr &Mp) {
    --Mp.MPtr;
    return Mp;
  }
  friend multi_ptr operator--(multi_ptr &Mp, int) {
    multi_ptr Tmp{Mp};
    --Mp.MPtr;
    return Tmp;
  }
  friend multi_ptr &operator+=(multi_ptr &Lhs, difference_type R) {
    Lhs.MPtr += R;
    return Lhs;
  }
  friend multi_ptr &operator-=(multi_ptr &Lhs, difference_type R) {
    Lhs.MPtr -= R;
    return Lhs;
  }
  friend multi_ptr operator+(const multi_ptr &Lhs, difference_type R) {
    return multi_ptr{Lhs.MPtr + R};
  }
  friend multi_ptr operator-(const multi_ptr &Lhs, difference_type R) {
    return multi_ptr{Lhs.MPtr - R};
  }

  _LIBSYCL_MULTI_PTR_REL_OPS(pointer_t)

private:
  pointer_t MPtr = nullptr;

  template <typename, access::address_space, access::decorated>
  friend class multi_ptr;
};

/// Specialization of the deprecated interface for void.
template <access::address_space Space>
class [[deprecated]] multi_ptr<void, Space, access::decorated::legacy> {
public:
  using value_type = void;
  using element_type = void;
  using difference_type = std::ptrdiff_t;

  using pointer_t = multi_ptr<void, Space, access::decorated::yes>::pointer;
  using const_pointer_t =
      multi_ptr<const void, Space, access::decorated::yes>::pointer;

  static constexpr access::address_space address_space = Space;

  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  ~multi_ptr() = default;

#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr(pointer_t Ptr) : MPtr(Ptr) {}
#endif

  multi_ptr(void *Ptr) : MPtr(detail::dynamicAddressCast<Space>(Ptr)) {}

  multi_ptr(std::nullptr_t) : MPtr(nullptr) {}

  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

#ifdef __SYCL_DEVICE_ONLY__
  // In a host compilation pointer_t is the same type as void *, so this
  // assignment would be a duplicate of the one below.
  multi_ptr &operator=(pointer_t Ptr) {
    MPtr = Ptr;
    return *this;
  }
#endif

  multi_ptr &operator=(void *Ptr) {
    MPtr = detail::dynamicAddressCast<Space>(Ptr);
    return *this;
  }

  multi_ptr &operator=(std::nullptr_t) {
    MPtr = nullptr;
    return *this;
  }

  pointer_t get() const { return MPtr; }
  std::add_pointer_t<value_type> get_raw() const { return MPtr; }
  pointer_t get_decorated() const { return MPtr; }

  operator void *() const { return MPtr; }

  /// Implicit conversion to a multi_ptr<const void>.
  operator multi_ptr<const void, Space, access::decorated::legacy>() const {
    using DstT = detail::DecoratedType_t<const void, Space> *;
    return multi_ptr<const void, Space, access::decorated::legacy>{
        static_cast<DstT>(MPtr)};
  }

  /// Explicit conversion to a multi_ptr of a concrete element type.
  template <typename ElementType>
  explicit
  operator multi_ptr<ElementType, Space, access::decorated::legacy>() const {
    using DstT = detail::DecoratedType_t<ElementType, Space> *;
    return multi_ptr<ElementType, Space, access::decorated::legacy>{
        reinterpret_cast<DstT>(MPtr)};
  }

  _LIBSYCL_MULTI_PTR_REL_OPS(pointer_t)

private:
  pointer_t MPtr = nullptr;

  template <typename, access::address_space, access::decorated>
  friend class multi_ptr;
};

/// Specialization of the deprecated interface for const void.
template <access::address_space Space>
class [[deprecated]] multi_ptr<const void, Space, access::decorated::legacy> {
public:
  using value_type = const void;
  using element_type = const void;
  using difference_type = std::ptrdiff_t;

  using pointer_t =
      multi_ptr<const void, Space, access::decorated::yes>::pointer;
  using const_pointer_t =
      multi_ptr<const void, Space, access::decorated::yes>::pointer;

  static constexpr access::address_space address_space = Space;

  multi_ptr() = default;
  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;
  ~multi_ptr() = default;

#ifdef __SYCL_DEVICE_ONLY__
  multi_ptr(pointer_t Ptr) : MPtr(Ptr) {}
#endif

  multi_ptr(const void *Ptr) : MPtr(detail::dynamicAddressCast<Space>(Ptr)) {}

  multi_ptr(std::nullptr_t) : MPtr(nullptr) {}

  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

#ifdef __SYCL_DEVICE_ONLY__
  // In a host compilation pointer_t is the same type as const void *, so this
  // assignment would be a duplicate of the one below.
  multi_ptr &operator=(pointer_t Ptr) {
    MPtr = Ptr;
    return *this;
  }
#endif

  multi_ptr &operator=(const void *Ptr) {
    MPtr = detail::dynamicAddressCast<Space>(Ptr);
    return *this;
  }

  multi_ptr &operator=(std::nullptr_t) {
    MPtr = nullptr;
    return *this;
  }

  pointer_t get() const { return MPtr; }
  std::add_pointer_t<value_type> get_raw() const { return MPtr; }
  pointer_t get_decorated() const { return MPtr; }

  operator const void *() const { return MPtr; }

  /// Explicit conversion to a multi_ptr of a concrete const element type.
  template <typename ElementType,
            std::enable_if_t<std::is_const_v<ElementType>, bool> = true>
  explicit
  operator multi_ptr<ElementType, Space, access::decorated::legacy>() const {
    using DstT = detail::DecoratedType_t<ElementType, Space> *;
    return multi_ptr<ElementType, Space, access::decorated::legacy>{
        reinterpret_cast<DstT>(MPtr)};
  }

  _LIBSYCL_MULTI_PTR_REL_OPS(pointer_t)

private:
  pointer_t MPtr = nullptr;

  template <typename, access::address_space, access::decorated>
  friend class multi_ptr;
};

#undef _LIBSYCL_MULTI_PTR_REL_OP
#undef _LIBSYCL_MULTI_PTR_REL_OPS

// SYCL 2020 4.7.7.2. Explicit pointer aliases.
//
// These default to access::decorated::legacy, so they name the deprecated
// interface unless a decoration is given explicitly. raw_* and decorated_*
// below name the current one.

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using global_ptr =
    multi_ptr<ElementType, access::address_space::global_space, IsDecorated>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using local_ptr =
    multi_ptr<ElementType, access::address_space::local_space, IsDecorated>;

// Deprecated in SYCL 2020 together with the constant address space.
template <typename ElementType>
using constant_ptr =
    multi_ptr<ElementType, access::address_space::constant_space,
              access::decorated::legacy>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using private_ptr =
    multi_ptr<ElementType, access::address_space::private_space, IsDecorated>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using generic_ptr =
    multi_ptr<ElementType, access::address_space::generic_space, IsDecorated>;

// The interface exposes a non-decorated pointer while keeping the address
// space information internally.

template <typename ElementType>
using raw_global_ptr =
    multi_ptr<ElementType, access::address_space::global_space,
              access::decorated::no>;

template <typename ElementType>
using raw_local_ptr = multi_ptr<ElementType, access::address_space::local_space,
                                access::decorated::no>;

template <typename ElementType>
using raw_private_ptr =
    multi_ptr<ElementType, access::address_space::private_space,
              access::decorated::no>;

template <typename ElementType>
using raw_generic_ptr =
    multi_ptr<ElementType, access::address_space::generic_space,
              access::decorated::no>;

// The interface exposes a decorated pointer.

template <typename ElementType>
using decorated_global_ptr =
    multi_ptr<ElementType, access::address_space::global_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_local_ptr =
    multi_ptr<ElementType, access::address_space::local_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_private_ptr =
    multi_ptr<ElementType, access::address_space::private_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_generic_ptr =
    multi_ptr<ElementType, access::address_space::generic_space,
              access::decorated::yes>;

/// \return a multi_ptr in address space Space created from Ptr. The result
/// is nullptr if Ptr does not point into Space.
template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
multi_ptr<ElementType, Space, DecorateAddress>
address_space_cast(ElementType *Ptr) {
  return multi_ptr<ElementType, Space, DecorateAddress>{
      detail::dynamicAddressCast<Space>(Ptr)};
}

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
__SYCL2020_DEPRECATED("use address_space_cast instead")
multi_ptr<ElementType, Space, DecorateAddress> make_ptr(ElementType *Ptr) {
  return address_space_cast<Space, DecorateAddress>(Ptr);
}

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_MULTI_PTR_HPP
