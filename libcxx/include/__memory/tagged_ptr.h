// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TAGGED_PTR_H
#define _LIBCPP___TAGGED_PTR_H

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 26
  
#include <__config>
#include <__type_traits/is_trivially_copyable.h>
#include <__assert>
#include <__bit/has_single_bit.h>
#include <__type_traits/rank.h>
#include <__memory/pointer_traits.h>
#include <compare>

_LIBCPP_BEGIN_NAMESPACE_STD

template <typename T, typename Y> concept convertible_to_from = std::convertible_to<Y, T> && std::convertible_to<T, Y>;

template <typename T> concept pointer_tagging_schema = 
requires {
  typename T::clean_pointer;
  typename T::dirty_pointer;
  typename T::tag_type;
} && requires(T::dirty_pointer payload, T::clean_pointer clean, T::tag_type tag) {
  requires convertible_to_from<typename T::tag_type, uintptr_t>;
  requires std::is_pointer_v<typename T::clean_pointer>;
  
  { T::encode_pointer_with_tag(clean, tag) } noexcept -> std::same_as<typename T::dirty_pointer>;
  { T::recover_pointer(payload) } noexcept -> std::same_as<typename T::clean_pointer>;
  { T::recover_tag(payload) } noexcept -> std::same_as<typename T::tag_type>;
};

template <typename T> concept pointer_tagging_schema_with_aliasing = pointer_tagging_schema<T> && requires(T::dirty_pointer payload) {
  { T::recover_aliasing_pointer(payload) } noexcept -> std::same_as<typename T::clean_pointer>;
};

namespace memory {

// no-op schema so I can better explain how schemas work
struct no_tag {
  template <typename T, typename Tag> struct schema {
    using clean_pointer = T *;
    using dirty_pointer = void *;
    using tag_type = Tag;
    
    static constexpr uintptr_t _mask = 0u;

    _LIBCPP_ALWAYS_INLINE static constexpr dirty_pointer encode_pointer_with_tag(clean_pointer _ptr, tag_type) noexcept {
      return (dirty_pointer)_ptr;
    }
    _LIBCPP_ALWAYS_INLINE static constexpr clean_pointer recover_pointer(dirty_pointer _ptr) noexcept {
      return (clean_pointer)_ptr;
    }
    _LIBCPP_ALWAYS_INLINE static constexpr clean_pointer recover_aliasing_pointer(dirty_pointer _ptr) noexcept {
      return (clean_pointer)_ptr;
    }
    _LIBCPP_ALWAYS_INLINE static constexpr tag_type recover_tag(dirty_pointer) noexcept {
      return {};
    }
  };
};

// most basic schema for tagging
// it lets user to provide their own mask
template <uintptr_t Mask> struct bitmask_tag {
  template <typename T, typename Tag> struct schema {
    using clean_pointer = T *;
    using dirty_pointer = void *;
    using tag_type = Tag;
    
    static constexpr uintptr_t _mask = Mask;

    _LIBCPP_ALWAYS_INLINE static constexpr dirty_pointer encode_pointer_with_tag(clean_pointer _ptr, tag_type _value) noexcept {
#if __has_builtin(__builtin_tag_pointer_mask_or)
      return static_cast<dirty_pointer>(__builtin_tag_pointer_mask_or((void *)(_ptr), static_cast<uintptr_t>(_value), _mask));
#else
      return reinterpret_cast<dirty_pointer>((reinterpret_cast<uintptr_t>(_ptr) & static_cast<uintptr_t>(_mask)) | (static_cast<uintptr_t>(_value) & ~static_cast<uintptr_t>(_mask)));
#endif
    }
    _LIBCPP_ALWAYS_INLINE static constexpr clean_pointer recover_pointer(dirty_pointer _ptr) noexcept {
#if __has_builtin(__builtin_tag_pointer_mask)
      return static_cast<clean_pointer>(__builtin_tag_pointer_mask((void *)_ptr, ~_mask));
#else
      return reinterpret_cast<clean_pointer>(reinterpret_cast<uintptr_t>(_ptr) & ~static_cast<uintptr_t>(_mask));
#endif
    }
    _LIBCPP_ALWAYS_INLINE static constexpr tag_type recover_tag(dirty_pointer _ptr) noexcept {
#if __has_builtin(__builtin_tag_pointer_mask_as_int)
      return static_cast<tag_type>(__builtin_tag_pointer_mask_as_int((void *)_ptr, _mask));
#else
      return static_cast<tag_type>(reinterpret_cast<uintptr_t>(_ptr) & static_cast<uintptr_t>(_mask));
#endif
    }
  };
};

// schema which allows only pointer of custom provided minimal alignment 
// otherwise it behaves as custom mask schema
template <unsigned Alignment> struct custom_alignment_tag {
  static_assert(std::has_single_bit(Alignment), "alignment must be power of 2");
  static constexpr uintptr_t mask = static_cast<uintptr_t>(Alignment) - 1ull;
  
  template <typename T, typename Tag> struct schema: bitmask_tag<mask>::template schema<T, Tag> {
    using _underlying_schema = bitmask_tag<mask>::template schema<T, Tag>;
  
    using clean_pointer = _underlying_schema::clean_pointer;
    using dirty_pointer = _underlying_schema::dirty_pointer;
    using tag_type = _underlying_schema::tag_type;
    
    _LIBCPP_ALWAYS_INLINE static constexpr dirty_pointer encode_pointer_with_tag(clean_pointer _ptr, tag_type _value) noexcept {
#if __has_builtin(__builtin_is_aligned)
      _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(__builtin_is_aligned(_ptr, Alignment), "Pointer must be aligned by provided alignemt for tagging");
#else
      if !consteval {
        _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(reinterpret_cast<uintptr_t>(std::addressof(_ptr)) % Alignment == 0, "Pointer must be aligned by provided alignemt for tagging");
      }
#endif
      return _underlying_schema::encode_pointer_with_tag(_ptr, _value);
    }
    
    using _underlying_schema::recover_pointer;
    using _underlying_schema::recover_tag;
  };
};

// default scheme which gives only bits from alignment
struct alignment_low_bits_tag {
  template <typename T, typename Tag> using schema = typename custom_alignment_tag<alignof(T)>::template schema<T, Tag>;
};

// scheme which shifts bits to left by Bits bits and gives the space for tagging
template <unsigned Bits> struct left_shift_tag {
  static constexpr unsigned _shift = Bits;
  static constexpr uintptr_t _mask = (uintptr_t{1u} << _shift) - 1u;

  template <typename T, typename Tag> struct schema {
    using clean_pointer = T *;
    using dirty_pointer = void *;
    using tag_type = Tag;
    
    static constexpr uintptr_t _mask = left_shift_tag::_mask;

    _LIBCPP_ALWAYS_INLINE static constexpr dirty_pointer encode_pointer_with_tag(clean_pointer _ptr, tag_type _value) noexcept {
#if __has_builtin(__builtin_tag_pointer_shift_or)
      return static_cast<dirty_pointer>(__builtin_tag_pointer_shift_or((void *)(_ptr), (uintptr_t)_value, _shift));
#else
      return reinterpret_cast<dirty_pointer>((reinterpret_cast<uintptr_t>(_ptr) << _shift) | (static_cast<uintptr_t>(_value) & ((1ull << static_cast<uintptr_t>(_shift)) - 1ull)));
#endif
    }
    _LIBCPP_ALWAYS_INLINE static constexpr clean_pointer recover_pointer(dirty_pointer _ptr) noexcept {
#if __has_builtin(__builtin_tag_pointer_unshift)
      return static_cast<clean_pointer>(__builtin_tag_pointer_unshift((void *)_ptr, _shift));
#else
      return reinterpret_cast<clean_pointer>(reinterpret_cast<uintptr_t>(_ptr) >> _shift);
#endif
    }
    _LIBCPP_ALWAYS_INLINE static constexpr tag_type recover_tag(dirty_pointer _ptr) noexcept {
#if __has_builtin(__builtin_tag_pointer_mask_as_int)
      return static_cast<tag_type>(__builtin_tag_pointer_mask_as_int((void *)_ptr, _mask));
#else
      return static_cast<tag_type>(reinterpret_cast<uintptr_t>(_ptr) & static_cast<uintptr_t>(_mask));
#endif      
    }
  };
};

// scheme which shifts pointer to left by 8 bits and give this space as guaranteed space for tagging
struct low_byte_tag {
  template <typename T, typename Tag> using schema = typename left_shift_tag<8>::template schema<T, Tag>;
};

// this will give user access to upper byte of pointer on aarch64
// also it supports recovering aliasing pointer as no-op (fast-path)
struct upper_byte_tag {
  template <typename T> static constexpr unsigned _shift = sizeof(T *) * 8ull - 8ull;
  template <typename T> static constexpr uintptr_t _mask = 0b1111'1111ull << _shift<T>;
  
  template <typename T, typename Tag> struct schema: bitmask_tag<_mask<T>>::template schema<T, Tag> {
    using _underlying_schema = bitmask_tag<_mask<T>>::template schema<T, Tag>;
    
    using clean_pointer = _underlying_schema::clean_pointer;
    using dirty_pointer = _underlying_schema::dirty_pointer;
    using tag_type = _underlying_schema::tag_type;
  
    _LIBCPP_ALWAYS_INLINE static constexpr clean_pointer recover_aliasing_pointer(dirty_pointer _ptr) noexcept {
      return (clean_pointer)_ptr;
    }
    
    using _underlying_schema::encode_pointer_with_tag;
    using _underlying_schema::recover_pointer;
    using _underlying_schema::recover_tag;
  };
};

// improved version of previous aarch64 upper byte scheme
// with added shifting tag value into position, so the tag doesn't need to know about exact position
struct upper_byte_shifted_tag: upper_byte_tag { 
  template <typename T, typename Tag> struct schema: upper_byte_tag::template schema<T, uintptr_t> {
    using _underlying_schema = upper_byte_tag::template schema<T, uintptr_t>;
    
    using clean_pointer = _underlying_schema::clean_pointer;
    using dirty_pointer = _underlying_schema::dirty_pointer;
    using tag_type = Tag;
  
    _LIBCPP_ALWAYS_INLINE static constexpr dirty_pointer encode_pointer_with_tag(clean_pointer _ptr, tag_type _value) noexcept {
      return _underlying_schema::encode_pointer_with_tag(_ptr, static_cast<uintptr_t>(_value) << upper_byte_tag::_shift<T>);
    }
    _LIBCPP_ALWAYS_INLINE static constexpr tag_type recover_tag(dirty_pointer _ptr) noexcept {
      return static_cast<tag_type>(_underlying_schema::recover_tag(_ptr) >> upper_byte_tag::_shift<T>);
    }
    
    using _underlying_schema::recover_pointer;
    using _underlying_schema::recover_aliasing_pointer;
  };
};

template <unsigned Bits> struct bits_needed {
  template <typename _T, typename _Tag> struct schema {
    // we can automatically choose suitable schema
  };
};

}


// forward declaration
template <typename _T, typename _Tag, typename _Schema> class tagged_ptr;

struct already_tagged_t {
  consteval explicit already_tagged_t(int) noexcept { }
};

constexpr auto already_tagged = already_tagged_t{0};

template <typename _Schema2, typename _T, typename _Tag, typename _Schema> constexpr auto scheme_pointer_cast(tagged_ptr<_T, _Tag, _Schema> in) noexcept {
  return tagged_ptr<_T, _Tag, _Schema2>{in.pointer(), in.tag()};
}

template <typename _Y, typename _T, typename _Tag, typename _Schema> constexpr auto const_pointer_cast(tagged_ptr<_T, _Tag, _Schema> in) noexcept {
  // TODO we can just use native pointer here
  return tagged_ptr<_Y, _Tag, _Schema>{const_cast<_Y*>(in.pointer()), in.tag()};
}

template <typename _Y, typename _T, typename _Tag, typename _Schema> constexpr auto static_pointer_cast(tagged_ptr<_T, _Tag, _Schema> in) noexcept {
  return tagged_ptr<_Y, _Tag, _Schema>{static_cast<_Y*>(in.pointer()), in.tag()};
}

template <typename _Y, typename _T, typename _Tag, typename _Schema> constexpr auto dynamic_pointer_cast(tagged_ptr<_T, _Tag, _Schema> in) noexcept {
  return tagged_ptr<_Y, _Tag, _Schema>{dynamic_cast<_Y*>(in.pointer()), in.tag()};
}

template <typename _Y, typename _T, typename _Tag, typename _Schema> auto reinterpret_pointer_cast(tagged_ptr<_T, _Tag, _Schema> in) noexcept {
  return tagged_ptr<_Y, _Tag, _Schema>{reinterpret_cast<_Y*>(in.pointer()), in.tag()};
}


// wrapper class containing the pointer value and provides access
template <typename _T, typename _Tag, typename _Schema> class tagged_ptr {
public:
  using schema = typename _Schema::template schema<_T, _Tag>;
  using dirty_pointer = typename schema::dirty_pointer;
  using clean_pointer = typename schema::clean_pointer;
  using tag_type = typename schema::tag_type;
  
  using element_type = typename std::pointer_traits<clean_pointer>::element_type;
  using difference_type = typename std::pointer_traits<clean_pointer>::difference_type;
  
  
  template <typename _Y> using rebind = tagged_ptr<_Y, _Tag, _Schema>;
  
private:
  
  dirty_pointer _pointer{nullptr};
  
  template <typename _Y, typename _T2, typename _Tag2, typename _Schema2> constexpr auto const_pointer_cast(tagged_ptr<_T2, _Tag2, _Schema2> in) noexcept -> rebind<_T>;
  
public:
  tagged_ptr() = default;
  constexpr tagged_ptr(nullptr_t) noexcept: _pointer{nullptr} { }
  tagged_ptr(const tagged_ptr &) = default;
  tagged_ptr(tagged_ptr &&) = default;
  ~tagged_ptr() = default;
  tagged_ptr & operator=(const tagged_ptr &) = default;
  tagged_ptr & operator=(tagged_ptr &&) = default;
  
  explicit constexpr tagged_ptr(already_tagged_t, dirty_pointer _ptr) noexcept: _pointer{_ptr} { }
  
  _LIBCPP_ALWAYS_INLINE explicit constexpr tagged_ptr(clean_pointer _ptr, tag_type _tag = {}) noexcept: _pointer{schema::encode_pointer_with_tag(_ptr, _tag)} {
    _LIBCPP_ASSERT_SEMANTIC_REQUIREMENT(pointer() == _ptr, "pointer must be recoverable after untagging");
    _LIBCPP_ASSERT_SEMANTIC_REQUIREMENT(tag() == _tag, "stored tag must be recoverable and within schema provided bit capacity");
  } 

  // accessors
  _LIBCPP_ALWAYS_INLINE constexpr decltype(auto) operator*() const noexcept {
    return *pointer();
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr clean_pointer operator->() const noexcept {
    return pointer();
  }
   
  template <typename...Ts> _LIBCPP_ALWAYS_INLINE constexpr decltype(auto) operator[](Ts... args) const noexcept requires std::is_array_v<element_type> && (sizeof...(Ts) == std::rank_v<element_type>) {
    return (*pointer())[args...];
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr decltype(auto) operator[](difference_type diff) const noexcept requires (!std::is_array_v<element_type>) {
    return *(pointer() + diff);
  }
  
  // swap
  _LIBCPP_ALWAYS_INLINE friend constexpr void swap(tagged_ptr & lhs, tagged_ptr & rhs) noexcept {
    std::swap(lhs._pointer, rhs._pointer);
  }
  
  // modifiers for tag
  _LIBCPP_ALWAYS_INLINE constexpr auto & set(tag_type new_tag) noexcept {
    // this is here so I can avoid checks
    // TODO we should be able to check what bits available
    _pointer = schema::encode_pointer_with_tag(pointer(), new_tag);
    return *this;
  }

  // modifiers for pointer
  _LIBCPP_ALWAYS_INLINE constexpr auto & operator++() noexcept {
    _pointer = tagged_ptr{pointer()+1u, tag()}._pointer;
    return *this;
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr auto operator++(int) noexcept {
    auto copy = auto(*this);
    this->operator++();
    return copy;
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr auto & operator+=(difference_type diff) noexcept {
    _pointer = tagged_ptr{pointer()+diff, tag()}._pointer;
    return *this;
  }
  
  _LIBCPP_ALWAYS_INLINE friend constexpr auto operator+(tagged_ptr lhs, difference_type diff) noexcept {
    lhs += diff;
    return lhs;
  }
  
  _LIBCPP_ALWAYS_INLINE friend constexpr auto operator+(difference_type diff, tagged_ptr rhs) noexcept {
    rhs += diff;
    return rhs;
  }
  
  _LIBCPP_ALWAYS_INLINE friend constexpr auto operator-(tagged_ptr lhs, difference_type diff) noexcept {
    lhs -= diff;
    return lhs;
  }
  
  _LIBCPP_ALWAYS_INLINE friend constexpr auto operator-(difference_type diff, tagged_ptr rhs) noexcept {
    rhs -= diff;
    return rhs;
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr auto & operator-=(difference_type diff) noexcept {
    _pointer = tagged_ptr{pointer()-diff, tag()}._pointer;
    return *this;
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr auto & operator--() noexcept {
    _pointer = tagged_ptr{pointer()-1u, tag()}._pointer;
    return *this;
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr auto operator--(int) noexcept {
    auto copy = auto(*this);
    this->operator--();
    return copy;
  }
  
  // observers
  constexpr dirty_pointer unsafe_dirty_pointer() const noexcept {
    // this function is not intentionally constexpr, as it is needed only to interact with
    // existing runtime code
    return _pointer;
  } 
  
  static constexpr bool support_aliasing_masking = pointer_tagging_schema_with_aliasing<schema>;
  
  _LIBCPP_ALWAYS_INLINE constexpr clean_pointer aliasing_pointer() const noexcept {
    if constexpr (support_aliasing_masking) {
      if !consteval {
        return schema::recover_aliasing_pointer(_pointer);
      }
    }
    
    return schema::recover_pointer(_pointer);
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr clean_pointer pointer() const noexcept {
    return schema::recover_pointer(_pointer);
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr tag_type tag() const noexcept {
    return schema::recover_tag(_pointer);
  }
  
  template <std::size_t I> [[nodiscard, clang::always_inline]] friend constexpr decltype(auto) get(tagged_ptr _pair) noexcept {
    static_assert(I < 3);
    if constexpr (I == 0) {
      return _pair.pointer();
    } else {
      return _pair.tag();
    }
  }
  
  _LIBCPP_ALWAYS_INLINE constexpr explicit operator bool() const noexcept {
    return pointer() != nullptr;
  }
  
  _LIBCPP_ALWAYS_INLINE friend constexpr ptrdiff_t operator-(tagged_ptr lhs, tagged_ptr rhs) noexcept {
    return lhs.pointer() - rhs.pointer();
  }
  
  // comparison operators
  _LIBCPP_ALWAYS_INLINE friend bool operator==(tagged_ptr, tagged_ptr) = default;
  
  struct _compare_object {
    clean_pointer pointer;
    tag_type tag;
    
    friend auto operator<=>(_compare_object, _compare_object) = default;
  };
  
  _LIBCPP_ALWAYS_INLINE friend constexpr auto operator<=>(tagged_ptr lhs, tagged_ptr rhs) noexcept {
    return _compare_object{lhs.pointer(), lhs.tag()} <=> _compare_object{rhs.pointer(), rhs.tag()};
  }
  _LIBCPP_ALWAYS_INLINE friend constexpr bool operator==(tagged_ptr lhs, clean_pointer rhs) noexcept {
    return lhs.pointer() == rhs;
  }
  _LIBCPP_ALWAYS_INLINE friend constexpr auto operator<=>(tagged_ptr lhs, clean_pointer rhs) noexcept {
    return lhs.pointer() <=> rhs;
  }
  _LIBCPP_ALWAYS_INLINE friend constexpr bool operator==(tagged_ptr lhs, nullptr_t) noexcept {
    return lhs.pointer() == nullptr;
  }
};

// to_address specialization
template <typename _T, typename _Tag, typename _Schema> static constexpr auto to_address(tagged_ptr<_T, _Tag, _Schema> p) noexcept -> tagged_ptr<_T, _Tag, _Schema>::element_type * {
  return p.pointer();
}

// iterator traits
template <typename _T, typename _Tag, typename _Schema>
struct _LIBCPP_TEMPLATE_VIS iterator_traits<tagged_ptr<_T, _Tag, _Schema>> {
  using _tagged_ptr = tagged_ptr<_T, _Tag, _Schema>;
  
  using iterator_category = std::random_access_iterator_tag;
  using iterator_concept = std::contiguous_iterator_tag;
  
  using value_type = _tagged_ptr::element_type;
  using reference = value_type &;
  using pointer = _tagged_ptr::clean_pointer;
  using difference_type = _tagged_ptr::difference_type;
};

// pointer traits
template <typename _T, typename _Tag, typename _Schema>
struct _LIBCPP_TEMPLATE_VIS pointer_traits<tagged_ptr<_T, _Tag, _Schema>> {
  using _tagged_ptr = tagged_ptr<_T, _Tag, _Schema>;
  using pointer = _tagged_ptr::clean_pointer;
  using element_type = _tagged_ptr::value_type;
  using difference_type = _tagged_ptr::difference_type;

  template <typename _Up> using rebind = typename _tagged_ptr::template rebind<_Up>;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr static _tagged_ptr pointer_to(pointer ptr) _NOEXCEPT {
    return _tagged_ptr{ptr};
  }
};

// support for tuple protocol so we can split tagged pointer to structured bindings:
// auto [ptr, tag] = tagged_ptr
template <typename _T, typename _Tag, typename _Schema>
struct tuple_size<tagged_ptr<_T, _Tag, _Schema>>: std::integral_constant<std::size_t, 2> {};

template <std::size_t I, typename _T, typename _Tag, typename _Schema>
struct tuple_element<I, tagged_ptr<_T, _Tag, _Schema>> {
  using _pair_type = tagged_ptr<_T, _Tag, _Schema>;
  using type = std::conditional_t<I == 0, typename _pair_type::clean_pointer, typename _pair_type::tag_type>;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

#endif // _LIBCPP___TAGGED_PTR_H
