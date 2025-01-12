// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___POINTER_TAG_PAIR_H
#define _LIBCPP___POINTER_TAG_PAIR_H

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 26

#include <__assert>
#include <__bit/bit_width.h>
#include <__bit/countr.h>
#include <__config>
#include <__tuple/tuple_element.h>
#include <__tuple/tuple_size.h>
#include <__type_traits/conditional.h>
#include <__type_traits/is_enum.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_object.h>
#include <__type_traits/make_unsigned.h>
#include <__type_traits/remove_cvref.h>
#include <__type_traits/underlying_type.h>
#include <__utility/swap.h>
#include <climits>
#include <compare>

_LIBCPP_BEGIN_NAMESPACE_STD

constexpr bool __is_ptr_aligned(const void* _ptr, size_t _alignment) noexcept {
#  if __has_builtin(__builtin_is_aligned)
  return __builtin_is_aligned(_ptr, _alignment);
#  else
  return reinterpret_cast<uintptr_t>(_ptr) % _alignment == 0;
#  endif
}

template <typename T>
struct __tag_underlying_type {
  using type = T;
};

template <typename T>
  requires(std::is_enum_v<T>)
struct __tag_underlying_type<T> {
  using type = std::underlying_type_t<T>;
};

template <typename T>
constexpr unsigned _alignment_bits_available = std::countr_zero(alignof(T));

template <typename T, unsigned Bits> struct _few_bits {
    T value : Bits;
};

template <typename T> struct _few_bits<T, 0> {
    struct zero {
        constexpr zero(T) noexcept { }
        constexpr operator T() const noexcept {
            return T{};
        }
    } value;
};

template <typename PointeeT, typename TagT, unsigned Bits = _alignment_bits_available<PointeeT>>
requires (std::is_object_v<PointeeT> && (std::is_integral_v<TagT> || std::is_enum_v<TagT>) && std::is_same_v<TagT, std::remove_cvref_t<TagT>>)
struct pointer_tag_pair {
public:
  using element_type          = PointeeT;
  using pointer_type          = PointeeT*;
  using tagged_pointer_type = void *; // or uintptr_t?
  using tag_type              = TagT;

private:
  static constexpr unsigned _alignment_needed = (1u << Bits);
  static constexpr uintptr_t _tag_mask        = (_alignment_needed - 1u);
  static constexpr uintptr_t _pointer_mask    = ~_tag_mask;

  using _underlaying_tag_type = typename __tag_underlying_type<tag_type>::type;
  // using unsigned_tag_type = std::make_unsigned_t<_underlaying_tag_type>;
  using unspecified_pointer_type = pointer_type;
  using _real_tag_type = _few_bits<_underlaying_tag_type, Bits>;

  unspecified_pointer_type _pointer{nullptr}; // required to have size same as sizeof(Pointee *)

  static_assert(Bits < sizeof(pointer_type) * CHAR_BIT);

  struct _tagged_t {};

  constexpr pointer_tag_pair(_tagged_t, unspecified_pointer_type _ptr) noexcept : _pointer{_ptr} {}

public:
  pointer_tag_pair() = default; // always noexcept
  constexpr pointer_tag_pair(nullptr_t) noexcept;
  pointer_tag_pair(const pointer_tag_pair&)            = default;
  pointer_tag_pair(pointer_tag_pair&&)                 = default;
  pointer_tag_pair& operator=(const pointer_tag_pair&) = default;
  pointer_tag_pair& operator=(pointer_tag_pair&&)      = default;

  // to store and tag pointer (only if eligible)
  // Precondition: bit_width(static_cast<make_usigned_t<U>>(tag)) <= Bits is true, where U is
  // underlying_type_t<TagType>> if is_enum_v<TagType> and TagType otherwise. Precondition: alignof(ptr) (there is
  // enough of bits) Mandates: alignment of element type >= bits requested This turn them into unsigned.
  constexpr pointer_tag_pair(pointer_type _ptr, tag_type _tag)
    requires(alignof(element_type) >= _alignment_needed)
  {
    const auto _native_tag = _real_tag_type{static_cast<_underlaying_tag_type>(_tag)}.value;
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        _native_tag == (static_cast<_underlaying_tag_type>(_tag)), "Tag value must fit into requested bits");
    //_LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(std::bit_width(_native_tag) <= Bits, "Tag type value must fits bits
    //requested");
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        std::__is_ptr_aligned(_ptr, _alignment_needed), "Pointer must be aligned by provided alignment for tagging");

    void* _tagged_ptr = __builtin_tag_pointer_mask_or((void*)_ptr, _native_tag, _tag_mask);
    _pointer          = static_cast<unspecified_pointer_type>(_tagged_ptr);
  }

  // Preconditions: p points to an object X of a type similar ([conv.qual]) to element_type, where X has alignment
  // byte_alignment  (inspired by aligned_accessor)
  //  The precondition needa to say null pointer value or the thing about pointing to object with aligment. ??
  template <unsigned _AlignmentPromised>
  static constexpr pointer_tag_pair from_overaligned(pointer_type _ptr, tag_type _tag) {
    const auto _native_tag = _real_tag_type{static_cast<_underlaying_tag_type>(_tag)}.value;
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        _native_tag == (static_cast<_underlaying_tag_type>(_tag)), "Tag value must fit into requested bits");
    //_LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(std::bit_width(static_cast<unsigned_tag_type>(_tag)) <= Bits, "Tag type
    //value must fits bits requested");
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        std::__is_ptr_aligned(_ptr, _AlignmentPromised), "Pointer must be aligned by provided alignment for tagging");

    void* _tagged_ptr = __builtin_tag_pointer_mask_or((void*)_ptr, _native_tag, _tag_mask);
    return pointer_tag_pair{_tagged_t{}, static_cast<unspecified_pointer_type>(_tagged_ptr)};
  }

  pointer_tag_pair from_tagged(tagged_pointer_type ptr) { // no-constexpr
    // Precondition: valid pointer if untagged??
    return pointer_tag_pair{_tagged_t{}, reinterpret_cast<unspecified_pointer_type>(ptr)};
  }

  // destructor
  ~pointer_tag_pair() = default;

  // accessors
  tagged_pointer_type tagged_pointer() const noexcept { 
    return reinterpret_cast<tagged_pointer_type>(_pointer); 
  }
  constexpr pointer_type pointer() const noexcept {
    return static_cast<pointer_type>(__builtin_tag_pointer_mask((void*)_pointer, _pointer_mask));
  }
  constexpr tag_type tag() const noexcept {
    const uintptr_t r = __builtin_tag_pointer_mask_as_int((void*)_pointer, _tag_mask);
    return static_cast<tag_type>(_real_tag_type{.value = static_cast<_underlaying_tag_type>(r)}.value);
  }

  // swap
  constexpr void swap(pointer_tag_pair& _rhs) noexcept { std::swap(_pointer, _rhs._pointer); }

  // comparing {pointer(), tag()} <=> {pointer(), tag()} for consistency
  friend constexpr auto operator<=>(pointer_tag_pair lhs, pointer_tag_pair rhs) noexcept {
    const auto _ptr_comp = lhs.pointer() <=> rhs.pointer();
    if (!std::is_eq(_ptr_comp)) {
      return _ptr_comp;
    }

    return lhs.tag() <=> rhs.tag();
  }
  friend bool operator==(pointer_tag_pair, pointer_tag_pair) = default;
};

// support for structured bindings
template <typename _Pointee, typename _Tag, unsigned _Bits>
struct tuple_size<pointer_tag_pair<_Pointee, _Tag, _Bits>> : std::integral_constant<std::size_t, 2> {};

template <typename _Pointee, typename _Tag, unsigned _Bits>
struct tuple_element<0, pointer_tag_pair<_Pointee, _Tag, _Bits>> {
  using type = _Pointee*;
};

template <typename _Pointee, typename _Tag, unsigned _Bits>
struct tuple_element<1, pointer_tag_pair<_Pointee, _Tag, _Bits>> {
  using type = _Tag;
};

template <typename _Pointee, typename _Tag, unsigned _Bits>
struct tuple_element<0, const pointer_tag_pair<_Pointee, _Tag, _Bits>> {
  using type = _Pointee*;
};

template <typename _Pointee, typename _Tag, unsigned _Bits>
struct tuple_element<1, const pointer_tag_pair<_Pointee, _Tag, _Bits>> {
  using type = _Tag;
};

// helpers

// std::get (with one overload as copying pointer_tag_pair is cheap)
template <size_t _I, typename _Pointee, typename _Tag, unsigned _Bits>
constexpr auto get(pointer_tag_pair<_Pointee, _Tag, _Bits> _pair) noexcept
    -> tuple_element<_I, pointer_tag_pair<_Pointee, _Tag, _Bits>>::type {
  if constexpr (_I == 0) {
    return _pair.pointer();
  } else {
    static_assert(_I == 1);
    return _pair.tag();
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

#endif // _LIBCPP___POINTER_TAG_PAIR_H