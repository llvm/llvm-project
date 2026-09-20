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

#include <__assert>
#include <__stacktrace/basic_stacktrace.h>
#include <__stacktrace/stacktrace_entry.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string_view>
#include <tuple>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct _Image {
  uintptr_t load_addr_{};
  uintptr_t slide_offset_{};
  string_view name_{}; // into the owning `_Images::names_` arena; see below
  bool is_main_prog_{};

  operator bool() const { return !name_.empty(); }

  bool operator<(_Image const& __rhs) const { return tuple{load_addr_, name_} < tuple{__rhs.load_addr_, __rhs.name_}; }
};

// Bump-allocating arena for image names.  These names are not permitted on caller's heap,
// and must survive after `stacktrace::current` returns, so allocate
struct _Names {
  constexpr static size_t __bytes = 32 << 10;
  char names_[__bytes]{};
  size_t used_{};

  static _Names instance_;

  // Copies `__name` into the shared arena and returns a view of the copy.
  // Best-effort; can return a truncated or empty view if out of space.
  string_view intern(const char* __name) {
    if (!__name) {
      return {};
    }
    size_t __len = std::min(strlen(__name), __bytes - used_);
    char* __dst  = names_ + used_;
    memcpy(__dst, __name, __len);
    used_ += __len;
    return {__dst, __len};
  }
};

// Contains _Image objects in sorted order, according to `_Image::operator<`.
struct _Images {
  // Includes two dummy low/high "sentinel" entries in addition to this max number of images
  constexpr static size_t __max_images = 256;
  std::array<_Image, __max_images + 2> images_{}; // includes left/right sentinels
  unsigned count_{};                              // image count, including sentinels
  std::mutex mutex_{};

  static _Images instance_;

  _Images() {
    images_[count_++] = {0uz, 0};  // sentinel at low end
    images_[count_++] = {~0uz, 0}; // sentinel at high end
  }

  // OS-specific: enumerate program images in this process's space. Defined as a weak no-op in
  // images.cpp, with the `*-impl.cpp` files providing strong definitions, where possible.
  void enumerate();

  _Image& operator[](size_t __index) {
    _LIBCPP_ASSERT(__index < count_, "__stacktrace::_Images::operator[]: index out of range");
    return images_[__index];
  }

  // Image representing the main program, or nullptr if we couldn't find it
  _Image* main_prog_image() {
    for (size_t __i = 1; __i < count_ - 1; __i++) {
      auto& __image = images_[__i];
      if (__image.is_main_prog_) {
        return &__image;
      }
    }
    return nullptr;
  }

  // Search the sorted images array for one containing this address.
  size_t find(uintptr_t __addr) const {
    auto __end = images_.begin() + count_;
    auto __it  = std::upper_bound(images_.begin(), __end, __addr, [](uintptr_t __a, _Image const& __img) {
      return __a < __img.load_addr_;
    });
    return size_t(__it - images_.begin()) - 1;
  }
};

void __populate_images(_Context& __cx);

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_IMAGES_H
