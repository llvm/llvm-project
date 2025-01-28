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

#  include <__assert>
#  include <__bit/bit_width.h>
#  include <__bit/countr.h>
#  include <__config>
#  include <__tuple/tuple_element.h>
#  include <__tuple/tuple_size.h>
#  include <__type_traits/conditional.h>
#  include <__type_traits/is_enum.h>
#  include <__type_traits/is_integral.h>
#  include <__type_traits/is_object.h>
#  include <__type_traits/make_unsigned.h>
#  include <__type_traits/remove_cvref.h>
#  include <__type_traits/underlying_type.h>
#  include <__utility/swap.h>
#  include <climits>
#  include <compare>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __impl {
constexpr bool __is_ptr_aligned(const void* _ptr, size_t _alignment) noexcept {
#  if __has_builtin(__builtin_is_aligned)
  return __builtin_is_aligned(_ptr, _alignment);
#  else
  return reinterpret_cast<uintptr_t>(_ptr) % _alignment == 0;
#  endif
}

template <typename T>
struct _underlying_type_or_identity {
  using type = T;
};

template <typename T>
  requires(std::is_enum_v<T>)
struct _underlying_type_or_identity<T> {
  using type = std::underlying_type_t<T>;
};

template <typename T>
using _underlying_type_or_identity_t = _underlying_type_or_identity<T>::type;
} // namespace __impl

template <typename PointeeT, typename TagT, unsigned Bits = std::countr_zero(alignof(PointeeT))>
  requires((std::is_void_v<PointeeT> || std::is_object_v<PointeeT>) && std::is_unsigned_v<__impl::_underlying_type_or_identity_t<TagT>> &&
           std::is_same_v<TagT, std::remove_cvref_t<TagT>>)
struct pointer_tag_pair {
public:
  using element_type        = PointeeT;
  using pointer_type        = PointeeT*;
  using tagged_pointer_type = void*; // I prefer `void *` to avoid roundtrip over an int and losing provenance
  using tag_type            = TagT;

private:
  static constexpr unsigned _alignment_needed = (1u << Bits);
  static constexpr uintptr_t _tag_mask        = (_alignment_needed - 1u);
  static constexpr uintptr_t _pointer_mask    = ~_tag_mask;

  // for clang it can be just `pointer_type`
  using _unspecified_pointer_type = pointer_type;
  _unspecified_pointer_type _pointer{nullptr}; // required to have size same as sizeof(Pointee *)

  // it doesn't make sense to ask for all or more bits than size of the pointer
  static_assert(Bits < sizeof(pointer_type) * CHAR_BIT);

  // internal constructor for `from_overaligned` and `from_tagged` functions
  struct _tagged_t {};
  constexpr pointer_tag_pair(_tagged_t, _unspecified_pointer_type _ptr) noexcept : _pointer{_ptr} {}

  static constexpr auto _pass_thru_mask(tag_type _tag) {
    auto _value = static_cast<__impl::_underlying_type_or_identity_t<tag_type>>(_tag) & _tag_mask;
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(static_cast<tag_type>(_value) == _tag,
                                          "Tag must fit requested bits");
    return _value;
  }

  static constexpr auto _from_bitfield_tag(std::size_t _tag) -> tag_type {
    return static_cast<tag_type>(static_cast<__impl::_underlying_type_or_identity_t<tag_type>>(_tag));
  }

public:
  // constructors
  pointer_tag_pair() = default; // always noexcept
  constexpr pointer_tag_pair(nullptr_t) noexcept: _pointer{nullptr} { }
  pointer_tag_pair(const pointer_tag_pair&)            = default;
  pointer_tag_pair(pointer_tag_pair&&)                 = default;
  pointer_tag_pair& operator=(const pointer_tag_pair&) = default;
  pointer_tag_pair& operator=(pointer_tag_pair&&)      = default;

  constexpr pointer_tag_pair(pointer_type _ptr, tag_type _tag)
    requires(alignof(element_type) >= _alignment_needed)
  {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(std::__impl::__is_ptr_aligned(_ptr, _alignment_needed),
                                          "Pointer must be aligned by provided alignment for tagging");

    _pointer = static_cast<_unspecified_pointer_type>(
        __builtin_tag_pointer_mask_or((void*)_ptr, _pass_thru_mask(_tag), _tag_mask));
  }
  
  // destructor
  ~pointer_tag_pair() = default;

  // special
  template <unsigned _AlignmentPromised>
  static constexpr pointer_tag_pair from_overaligned(pointer_type _ptr, tag_type _tag)
    requires(_AlignmentPromised >= _alignment_needed)
  {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(std::__impl::__is_ptr_aligned(_ptr, _AlignmentPromised),
                                          "Pointer must be aligned by provided alignment for tagging");

    return pointer_tag_pair{_tagged_t{},
                            static_cast<_unspecified_pointer_type>(
                                __builtin_tag_pointer_mask_or((void*)_ptr, _pass_thru_mask(_tag), _tag_mask))};
  }

  static pointer_tag_pair from_tagged(tagged_pointer_type ptr) { // no-constexpr
    // Precondition: valid pointer if untagged
    return pointer_tag_pair{_tagged_t{}, reinterpret_cast<_unspecified_pointer_type>(ptr)};
  }

  // accessors
  tagged_pointer_type tagged_pointer() const noexcept { return reinterpret_cast<tagged_pointer_type>(_pointer); }
  constexpr pointer_type pointer() const noexcept {
    return static_cast<pointer_type>(__builtin_tag_pointer_mask((void*)_pointer, _pointer_mask));
  }
  constexpr tag_type tag() const noexcept {
    return _from_bitfield_tag(__builtin_tag_pointer_mask_as_int((void*)_pointer, _tag_mask));
  }

  // swap
  friend constexpr void swap(pointer_tag_pair& _lhs, pointer_tag_pair& _rhs) noexcept { std::swap(_lhs._pointer, _rhs._pointer); }

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