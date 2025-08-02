// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H
#define _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H

#include <__algorithm/copy_n.h>
#include <__bit/bit_cast.h>
#include <__config>
#include <__cstddef/ptrdiff_t.h>
#include <__cstddef/size_t.h>
#include <__functional/hash.h>
#include <__iterator/iterator_traits.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/view_interface.h>
#include <__text_encoding/text_encoding_rep.h>
#include <cstdint>
#include <string_view>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

struct text_encoding {
  static constexpr size_t max_name_length = 63;
  using id                                = std::__text_encoding_rep::__id;
  using enum id;

  _LIBCPP_HIDE_FROM_ABI constexpr text_encoding() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit text_encoding(string_view __enc) noexcept
      : __encoding_rep_(__text_encoding_rep::__find_encoding_data(__enc)) {
    __enc.copy(__name_, max_name_length, 0);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr text_encoding(id __i) noexcept
      : __encoding_rep_(__text_encoding_rep::__find_encoding_data_by_id(__i)) {
    if (__encoding_rep_->__name_[0] != '\0')
      std::copy_n(__encoding_rep_->__name_, std::char_traits<char>::length(__encoding_rep_->__name_), __name_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr id mib() const noexcept { return id(__encoding_rep_->__mib_rep_); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const char* name() const noexcept { return __name_; }

  // [text.encoding.aliases], class text_encoding::aliases_view
  struct aliases_view : ranges::view_interface<aliases_view> {
    struct __end_sentinel {};
    struct __iterator {
      using value_type      = const char*;
      using reference       = const char*;
      using difference_type = ptrdiff_t;

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator() noexcept = default;

      _LIBCPP_HIDE_FROM_ABI constexpr value_type operator*() const {
        _LIBCPP_ASSERT(__can_dereference(), "Dereferencing invalid aliases_view iterator!");
        return __data_->__name_;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr value_type operator[](difference_type __n) const {
        auto __it = *this;
        return *(__it + __n);
      }

      _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(__iterator __it, difference_type __n) {
        __it += __n;
        return __it;
      }

      _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(difference_type __n, __iterator __it) {
        __it += __n;
        return __it;
      }

      _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(__iterator __it, difference_type __n) {
        __it -= __n;
        return __it;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr difference_type operator-(const __iterator& __other) const {
        _LIBCPP_ASSERT(__other.__mib_rep_ == __mib_rep_, "Subtracting ranges of two different text encodings!");
        return __data_ - __other.__data_;
      }

      _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(difference_type __n, __iterator& __it) {
        __it -= __n;
        return __it;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
        __data_++;
        return *this;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int) {
        auto __old = *this;
        __data_++;
        return __old;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--() {
        __data_--;
        return *this;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int) {
        auto __old = *this;
        __data_--;
        return __old;
      }

      // Check if going past the encoding data list array and if the new index has the same id, if not then
      // replace it with a sentinel "out-of-bounds" iterator.
      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __n) {
        if (__data_) {
          if (__n > 0) {
            if ((__data_ + __n) < std::end(__text_encoding_rep::__text_encoding_data) &&
                __data_[__n - 1].__mib_rep_ == __mib_rep_)
              __data_ += __n;
            else
              *this = __iterator{};
          } else if (__n < 0) {
            if ((__data_ + __n) > __text_encoding_rep::__text_encoding_data && __data_[__n].__mib_rep_ == __mib_rep_)
              __data_ += __n;
            else
              *this = __iterator{};
          }
        }
        return *this;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __n) { return operator+=(-__n); }

      _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const __iterator& __it) const {
        return __data_ == __it.__data_ && __it.__mib_rep_ == __mib_rep_;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(__end_sentinel) const { return !__can_dereference(); }

      _LIBCPP_HIDE_FROM_ABI constexpr auto operator<=>(__iterator __it) const { return __data_ <=> __it.__data_; }

    private:
      friend struct aliases_view;

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator(const __text_encoding_rep::__encoding_data* __enc_d) noexcept
          : __data_(__enc_d), __mib_rep_(__enc_d ? __enc_d->__mib_rep_ : 0) {}

      _LIBCPP_HIDE_FROM_ABI constexpr bool __can_dereference() const {
        return __data_ && __data_->__mib_rep_ == __mib_rep_;
      }

      // default iterator is a sentinel
      const __text_encoding_rep::__encoding_data* __data_ = nullptr;
      __text_encoding_rep::__id_rep __mib_rep_            = 0;
    };

    _LIBCPP_HIDE_FROM_ABI constexpr __iterator begin() const { return __iterator{__view_data_}; }
    _LIBCPP_HIDE_FROM_ABI constexpr __end_sentinel end() const { return __end_sentinel{}; }

  private:
    friend struct text_encoding;

    _LIBCPP_HIDE_FROM_ABI constexpr aliases_view(const __text_encoding_rep::__encoding_data* __d) : __view_data_(__d) {}
    const __text_encoding_rep::__encoding_data* __view_data_ = nullptr;
  };

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr aliases_view aliases() const {
    return __encoding_rep_->__name_[0] ? aliases_view(__encoding_rep_) : aliases_view(nullptr);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const text_encoding& __a, const text_encoding& __b) noexcept {
    if (__a.mib() == id::other && __b.mib() == id::other)
      return __text_encoding_rep::__comp_name(__a.__name_, __b.__name_);
    return __a.mib() == __b.mib();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const text_encoding& __encoding, id __i) noexcept {
    return __encoding.mib() == __i;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static consteval text_encoding literal() noexcept {
    // TODO: Remove this branch once we have __GNUC_EXECUTION_CHARSET_NAME or __clang_literal_encoding__ unconditionally
#  ifdef __GNUC_EXECUTION_CHARSET_NAME
    return text_encoding(__GNUC_EXECUTION_CHARSET_NAME);
#  elif defined(__clang_literal_encoding__)
    return text_encoding(__clang_literal_encoding__);
#  else
    return text_encoding();
#  endif
  }

#  if _LIBCPP_HAS_LOCALIZATION
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static text_encoding environment() {
    return std::bit_cast<text_encoding>(__text_encoding_rep::__get_locale_encoding(""));
  }

  template <id __i>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static bool environment_is() {
    return environment() == __i;
  }
#  else
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static text_encoding environment() = delete;
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static bool environment_is()       = delete;
#  endif // _LIBCPP_HAS_LOCALIZATION

private:
  const __text_encoding_rep::__encoding_data* __encoding_rep_ = __text_encoding_rep::__text_encoding_data + 1;
  char __name_[max_name_length + 1]                           = {0};
};

template <>
struct hash<text_encoding> {
  size_t operator()(const text_encoding& __enc) const noexcept {
    return std::hash<std::text_encoding::id>()(__enc.mib());
  }
};

template <>
inline constexpr bool ranges::enable_borrowed_range<text_encoding::aliases_view> = true;
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H
