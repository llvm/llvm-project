// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_IMAGES_H
#define _LIBCPP_STACKTRACE_IMAGES_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#include <__stacktrace/stacktrace_entry.h>
#include <array>
#include <cstdint>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct _Image;
struct _Images;

struct _Image {
  uintptr_t loaded_at_{};
  uintptr_t slide_{};
  char name_[__stacktrace::_Entry::__max_file_len]{0};
  bool is_main_prog_{};

  bool operator<(_Image const& __rhs) const {
    if (loaded_at_ < __rhs.loaded_at_) {
      return true;
    }
    if (loaded_at_ > __rhs.loaded_at_) {
      return false;
    }
    return strcmp(name_, __rhs.name_) < 0;
  }
  operator bool() const { return name_[0]; }
};

/**
 * Contains an array `images_`, which will include `prog_image` objects (see above)
 * collected in an OS-dependent way.  After construction these images will be sorted
 * according to their load address; there will also be two sentinels with dummy
 * addresses (0x0000... and 0xFFFF...) to simplify search functions.
 *
 * After construction, images_ and count_ look like:
 *  [0]             [1]             [2]             [3]       ...     [count_ - 1]
 *  (sentinel)      foo.exe         libc++so.1      libc.so.6         (sentinel)
 *  0x000000000000  0x000100000000  0x633b00000000  0x7c5500000000    0xffffffffffff
 */
struct _Images {
  constexpr static size_t k_max_images = 256;
  std::array<_Image, k_max_images + 2> images_{}; // space for the L/R sentinels
  unsigned count_{0};                             // image count, including sentinels

  /** An OS-specific constructor is defined. */
  _LIBCPP_EXPORTED_FROM_ABI _Images();

  /** Get prog_image by index (0 <= index < count_) */
  _Image& operator[](size_t __index) {
    _LIBCPP_ASSERT(__index < count_, "index out of range");
    return images_.at(__index);
  }

  /** Image representing the main program, or nullptr if we couldn't find it */
  _Image* main_prog_image() {
    for (size_t __i = 1; __i < count_ - 1; __i++) {
      auto& __image = images_[__i];
      if (__image.is_main_prog_) {
        return &__image;
      }
    }
    return nullptr;
  }

  /** Search the sorted images array for one containing this address. */
  void find(size_t* __index, uintptr_t __addr) {
    // `index` slides left/right as we search through images.
    // It's (probably) likely several consecutive entries are from the same image, so
    // each iteration's `find` uses the same starting point, making it (probably) constant-time.
    // XXX Is this more efficient in practice than e.g. `std::set` and `upper_bound`?
    if (*__index < 1) {
      *__index = 1;
    }
    if (*__index > count_ - 1) {
      *__index = count_ - 1;
    }
    while (images_[*__index]) {
      if (__addr < images_[*__index].loaded_at_) {
        --*__index;
      } else if (__addr >= images_[*__index + 1].loaded_at_) {
        ++*__index;
      } else {
        break;
      }
    }
  }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_IMAGES_H
