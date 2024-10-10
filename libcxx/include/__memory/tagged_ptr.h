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
#include "__bit/has_single_bit.h"
#include <__type_traits/rank.h>
#include "pointer_traits.h"
#include <compare>

_LIBCPP_BEGIN_NAMESPACE_STD

template <typename T, typename Y> concept convertible_to_from = std::convertible_to<Y, T> && std::convertible_to<T, Y>;
  
template <typename T> concept pointer_tagging_schema = requires(T::dirty_pointer payload, T::clean_pointer clean, T::tag_type tag) {
  //requires convertible_to_from<typename T::tag_type, uintptr_t>;
  requires std::is_pointer_v<typename T::clean_pointer>;
  
  { T::encode_pointer_with_tag(clean, tag) } noexcept -> std::same_as<typename T::dirty_pointer>;
  { T::recover_pointer(payload) } noexcept -> std::same_as<typename T::clean_pointer>;
  { T::recover_value(payload) } noexcept -> std::same_as<typename T::tag_type>;
};

template <typename T> concept pointer_tagging_schema_with_aliasing = pointer_tagging_schema<T> && requires(T::dirty_pointer payload) {
  { T::recover_aliasing_pointer(payload) } noexcept -> std::same_as<typename T::clean_pointer>;
};

struct no_tag {
  template <typename T, typename Tag> struct schema {
    using clean_pointer = T *;
    using dirty_pointer = void *;
    using tag_type = Tag;

    [[clang::always_inline]] static constexpr dirty_pointer encode_pointer_with_tag(clean_pointer _ptr, tag_type) noexcept {
      return (dirty_pointer)_ptr;
    }
    [[clang::always_inline]] static constexpr clean_pointer recover_pointer(dirty_pointer _ptr) noexcept {
      return (clean_pointer)_ptr;
    }
    [[clang::always_inline]] static constexpr tag_type recover_value(dirty_pointer) noexcept {
      return {};
    }
  };
};

template <uintptr_t Mask> struct bitmask_tag {
  static constexpr uintptr_t _mask = Mask;

  template <typename T, typename Tag> struct schema {
    using clean_pointer = T *;
    using dirty_pointer = void *;
    using tag_type = Tag;

    [[clang::always_inline]] static constexpr dirty_pointer encode_pointer_with_tag(clean_pointer _ptr, tag_type _value) noexcept {
      return static_cast<dirty_pointer>(__builtin_tag_pointer_mask_or((void *)(_ptr), static_cast<uintptr_t>(_value), _mask));
    }
    [[clang::always_inline]] static constexpr clean_pointer recover_pointer(dirty_pointer _ptr) noexcept {
      return static_cast<clean_pointer>(__builtin_tag_pointer_mask((void *)_ptr, ~_mask));
    }
    [[clang::always_inline]] static constexpr tag_type recover_value(dirty_pointer _ptr) noexcept {
      return static_cast<tag_type>(__builtin_tag_pointer_mask_as_int((void *)_ptr, _mask));
    }
  };
};

template <unsigned Alignment> struct custom_alignment_tag {
  static constexpr uintptr_t mask = (static_cast<uintptr_t>(1u) << static_cast<uintptr_t>(Alignment)) - 1ull;
  template <typename T, typename Tag> using schema = typename bitmask_tag<mask>::template schema<T, Tag>;
};

struct alignment_low_bits_tag {
  template <typename T> static constexpr unsigned alignment = alignof(T);
  template <typename T, typename Tag> using schema = typename custom_alignment_tag<alignment<T>>::template schema<T, Tag>;
};

template <unsigned Bits> struct shift_tag {
  static constexpr unsigned _shift = Bits;
  static constexpr uintptr_t _mask = (uintptr_t{1u} << _shift) - 1u;

  template <typename T, typename Tag> struct schema {
    using clean_pointer = T *;
    using dirty_pointer = void *;
    using tag_type = Tag;

    [[clang::always_inline]] static constexpr dirty_pointer encode_pointer_with_tag(clean_pointer _ptr, tag_type _value) noexcept {
      return static_cast<dirty_pointer>(__builtin_tag_pointer_shift_or((void *)(_ptr), (uintptr_t)_value, _shift));
    }
    [[clang::always_inline]] static constexpr clean_pointer recover_pointer(dirty_pointer _ptr) noexcept {
      return static_cast<clean_pointer>(__builtin_tag_pointer_unshift((void *)_ptr, _shift));
    }
    [[clang::always_inline]] static constexpr tag_type recover_value(dirty_pointer _ptr) noexcept {
      return static_cast<tag_type>(__builtin_tag_pointer_mask_as_int((void *)_ptr, _mask));
    }
  };
};

struct low_byte_tag {
  template <typename T, typename Tag> using schema = typename shift_tag<8>::template schema<T, Tag>;
};

struct upper_byte_tag {
  template <typename T> static constexpr unsigned _shift = sizeof(T *) * 8ull - 8ull;
  template <typename T> static constexpr uintptr_t _mask = 0b1111'1111ull << _shift<T>;
  
  template <typename T, typename Tag> using schema = typename bitmask_tag<_mask<T>>::template schema<T, Tag>;
};

struct upper_byte_shifted_tag: upper_byte_tag {
  template <typename T, typename Tag> struct schema {
    using _underlying_schema = typename upper_byte_tag::template schema<T, uintptr_t>;
    static constexpr unsigned _shift = upper_byte_tag::template _shift<T>;
    
    using clean_pointer = T *;
    using dirty_pointer = void *;
    using tag_type = Tag;
  
    [[clang::always_inline]] static constexpr dirty_pointer encode_pointer_with_tag(clean_pointer _ptr, tag_type _value) noexcept {
      return _underlying_schema::encode_pointer_with_tag(_ptr, static_cast<uintptr_t>(_value) << _shift);
    }
    [[clang::always_inline]] static constexpr clean_pointer recover_pointer(dirty_pointer _ptr) noexcept {
      return _underlying_schema::recover_pointer(_ptr);
    }
    [[clang::always_inline]] static constexpr tag_type recover_value(dirty_pointer _ptr) noexcept {
      return static_cast<tag_type>(_underlying_schema::recover_value(_ptr) >> _shift);
    }
  };
};



// forward declaration
template <typename _T, typename _Tag = uintptr_t, typename _Schema = alignment_low_bits_tag> class tagged_ptr;


template <typename _Schema, typename _T, typename _Tag = uintptr_t> constexpr auto tag_ptr(_T * _ptr, _Tag _tag = {}) noexcept {
  return tagged_ptr<_T, _Tag, _Schema>{_ptr, _tag};
}

template <typename _T, typename _Tag, typename _Schema = alignment_low_bits_tag> constexpr auto tagged_pointer_cast(typename _Schema::template schema<_T, _Tag>::dirty_pointer _ptr) noexcept -> tagged_ptr<_T, _Tag, _Schema> {
  using result_type = tagged_ptr<_T, _Tag, _Schema>;
  return result_type{typename result_type::already_tagged_tag{_ptr}};
}

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
  
  using value_type = std::remove_cvref_t<decltype(*std::declval<clean_pointer>())>;
  using difference_type = typename std::pointer_traits<clean_pointer>::difference_type;
  
  
  template <typename _Y> using rebind = tagged_ptr<_Y, _Tag, _Schema>;
  
private:
  
  dirty_pointer _pointer{nullptr};
  
  friend constexpr auto tagged_pointer_cast<_T, _Tag, _Schema>(typename _Schema::template schema<_T, _Tag>::dirty_pointer ptr) noexcept -> tagged_ptr<_T, _Tag, _Schema>;
  
  struct already_tagged_tag {
    dirty_pointer _ptr;
  };
 
  // special hidden constructor to allow constructing unsafely
  [[clang::always_inline]] constexpr tagged_ptr(already_tagged_tag _in) noexcept: _pointer{_in._ptr} { }
  
  template <typename _Y, typename _T2, typename _Tag2, typename _Schema2> constexpr auto const_pointer_cast(tagged_ptr<_T2, _Tag2, _Schema2> in) noexcept -> rebind<_T>;
  
public:
  tagged_ptr() = default;
  consteval tagged_ptr(nullptr_t) noexcept: _pointer{nullptr} { }
  tagged_ptr(const tagged_ptr &) = default;
  tagged_ptr(tagged_ptr &&) = default;
  ~tagged_ptr() = default;
  tagged_ptr & operator=(const tagged_ptr &) = default;
  tagged_ptr & operator=(tagged_ptr &&) = default;
  
  [[clang::always_inline]] explicit constexpr tagged_ptr(clean_pointer _ptr, tag_type _tag = {}) noexcept: _pointer{schema::encode_pointer_with_tag(_ptr, _tag)} {
    _LIBCPP_ASSERT_SEMANTIC_REQUIREMENT(pointer() == _ptr, "pointer must be recoverable after untagging");
    _LIBCPP_ASSERT_SEMANTIC_REQUIREMENT(tag() == _tag, "stored tag must be recoverable and within schema provided bit capacity");
  } 

  // accessors
  [[clang::always_inline]] constexpr decltype(auto) operator*() const noexcept {
    return *pointer();
  }
  
  [[clang::always_inline]] constexpr clean_pointer operator->() const noexcept {
    return pointer();
  }
   
  template <typename...Ts> [[clang::always_inline]] [[clang::always_inline]] constexpr decltype(auto) operator[](Ts... args) const noexcept requires std::is_array_v<value_type> && (sizeof...(Ts) == std::rank_v<value_type>) {
    return (*pointer())[args...];
  }
  
  [[clang::always_inline]] constexpr decltype(auto) operator[](difference_type diff) const noexcept requires (!std::is_array_v<value_type>) {
    return *(pointer() + diff);
  }
  
  // swap
  [[clang::always_inline]] friend constexpr void swap(tagged_ptr & lhs, tagged_ptr & rhs) noexcept {
    std::swap(lhs._pointer, rhs._pointer);
  }
  
  // modifiers for tag
  [[clang::always_inline]] constexpr auto & set(tag_type new_tag) noexcept {
    // this is here so I can avoid checks
    // TODO we should be able to check what bits available
    _pointer = schema::encode_pointer_with_tag(pointer(), new_tag);
    return *this;
  }
  
  [[clang::always_inline]] constexpr auto & set_union(tag_type addition) noexcept {
    return set(tag() | addition);
  }
  
  [[clang::always_inline]] constexpr auto & set_difference(tag_type mask) noexcept {
    return set(tag() & (~static_cast<uintptr_t>(mask)));
  }
  
  [[clang::always_inline]] constexpr auto & set_intersection(tag_type mask) noexcept {
    return set(tag() & mask);
  }
  
  [[clang::always_inline]] constexpr auto & set_all() noexcept {
    return set(static_cast<tag_type>(0xFFFFFFFF'FFFFFFFFull));
  }

  // modifiers for pointer
  [[clang::always_inline]] constexpr auto & operator++() noexcept {
    _pointer = tagged_ptr{pointer()+1u, tag()}._pointer;
    return *this;
  }
  
  [[clang::always_inline]] constexpr auto operator++(int) noexcept {
    auto copy = auto(*this);
    this->operator++();
    return copy;
  }
  
  [[clang::always_inline]] constexpr auto & operator+=(difference_type diff) noexcept {
    _pointer = tagged_ptr{pointer()+diff, tag()}._pointer;
    return *this;
  }
  
  [[clang::always_inline]] friend constexpr auto operator+(tagged_ptr lhs, difference_type diff) noexcept {
    lhs += diff;
    return lhs;
  }
  
  [[clang::always_inline]] friend constexpr auto operator+(difference_type diff, tagged_ptr rhs) noexcept {
    rhs += diff;
    return rhs;
  }
  
  [[clang::always_inline]] friend constexpr auto operator-(tagged_ptr lhs, difference_type diff) noexcept {
    lhs -= diff;
    return lhs;
  }
  
  [[clang::always_inline]] friend constexpr auto operator-(difference_type diff, tagged_ptr rhs) noexcept {
    rhs -= diff;
    return rhs;
  }
  
  [[clang::always_inline]] constexpr auto & operator-=(difference_type diff) noexcept {
    _pointer = tagged_ptr{pointer()-diff, tag()}._pointer;
    return *this;
  }
  
  [[clang::always_inline]] constexpr auto & operator--() noexcept {
    _pointer = tagged_ptr{pointer()-1u, tag()}._pointer;
    return *this;
  }
  
  [[clang::always_inline]] constexpr auto operator--(int) noexcept {
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
  
  [[clang::always_inline]] constexpr clean_pointer aliasing_pointer() const noexcept {
    if constexpr (support_aliasing_masking) {
      if !consteval {
        return schema::recover_aliasing_pointer(_pointer);
      }
    }
    
    return schema::recover_pointer(_pointer);
  }
  
  [[clang::always_inline]] constexpr clean_pointer pointer() const noexcept {
    return schema::recover_pointer(_pointer);
  }
  
  [[clang::always_inline]] constexpr tag_type tag() const noexcept {
    return schema::recover_value(_pointer);
  }
  
  template <std::size_t I> [[nodiscard, clang::always_inline]] friend constexpr decltype(auto) get(tagged_ptr _pair) noexcept {
    static_assert(I < 3);
    if constexpr (I == 0) {
      return _pair.pointer();
    } else {
      return _pair.tag();
    }
  }
  
  [[clang::always_inline]] constexpr explicit operator bool() const noexcept {
    return pointer() != nullptr;
  }
  
  [[clang::always_inline]] friend constexpr ptrdiff_t operator-(tagged_ptr lhs, tagged_ptr rhs) noexcept {
    return lhs.pointer() - rhs.pointer();
  }
  
  // comparison operators
  [[clang::always_inline]] friend bool operator==(tagged_ptr, tagged_ptr) = default;
  
  struct _compare_object {
    clean_pointer pointer;
    tag_type tag;
    
    friend auto operator<=>(_compare_object, _compare_object) = default;
  };
  
  [[clang::always_inline]] friend constexpr auto operator<=>(tagged_ptr lhs, tagged_ptr rhs) noexcept {
    return _compare_object{lhs.pointer(), lhs.tag()} <=> _compare_object{rhs.pointer(), rhs.tag()};
  }
  [[clang::always_inline]] friend constexpr bool operator==(tagged_ptr lhs, clean_pointer rhs) noexcept {
    return lhs.pointer() == rhs;
  }
  [[clang::always_inline]] friend constexpr auto operator<=>(tagged_ptr lhs, clean_pointer rhs) noexcept {
    return lhs.pointer() <=> rhs;
  }
  [[clang::always_inline]] friend constexpr bool operator==(tagged_ptr lhs, nullptr_t) noexcept {
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
  
  using value_type = _tagged_ptr::value_type;
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
  
  // what to do with this?
  template <typename _Up> using rebind = typename _tagged_ptr::template rebind<_Up>;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr static pointer pointer_to(pointer ptr) _NOEXCEPT {
    return _tagged_ptr{ptr};
  }
};

// we are defaulting always to low_bits schema
template <typename _T> tagged_ptr(_T *) -> tagged_ptr<_T>;
template <typename _T, typename _Tag> tagged_ptr(_T *, _Tag) -> tagged_ptr<_T, _Tag>;

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
