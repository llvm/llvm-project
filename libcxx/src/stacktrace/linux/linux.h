//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_LINUX_H
#define _LIBCPP_STACKTRACE_LINUX_H

#include "../common/config.h"

#if defined(_LIBCPP_STACKTRACE_LINUX)

#  include <algorithm>
#  include <array>
#  include <cassert>
#  include <cstddef>
#  include <cstdlib>
#  include <link.h>
#  include <string_view>
#  include <unistd.h>

#  include "../common/images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct context;

struct linux {
  context& cx_;
  void ident_modules();
  void symbolize();

private:
  void resolve_main_elf_syms(std::string_view elf_name);
};

struct images {
  // How many images this contains, including the left/right sentinels.
  unsigned count_{0};
  std::array<image, k_max_images + 2> images_{};

  int add(dl_phdr_info& info) {
    assert(count_ < k_max_images);
    auto isFirst        = (count_ == 0);
    auto& image         = images_.at(count_++);
    image.loaded_at_    = info.dlpi_addr;
    image.slide_        = info.dlpi_addr;
    image.name_         = info.dlpi_name;
    image.is_main_prog_ = isFirst; // first one is the main ELF
    if (image.name_.empty() && isFirst) {
      static char buffer[PATH_MAX + 1];
      uint32_t length = sizeof(buffer);
      if (readlink("/proc/self/exe", buffer, length) > 0) {
        image.name_ = buffer;
      }
    }
    return count_ == k_max_images; // return nonzero if we're at the limit
  }

  static int callback(dl_phdr_info* info, size_t, void* self) { return (*(images*)(self)).add(*info); }

  images() {
    dl_iterate_phdr(images::callback, this);
    images_[count_++] = {0uz, 0};  // sentinel at low end
    images_[count_++] = {~0uz, 0}; // sentinel at high end
    std::sort(images_.begin(), images_.begin() + count_ - 1);
  }

  image& operator[](size_t index) {
    assert(index < count_);
    return images_.at(index);
  }

  image* mainProg() {
    for (auto& image : images_) {
      if (image.is_main_prog_) {
        return &image;
      }
    }
    return nullptr;
  }

  static images& get() {
    static images images;
    return images;
  }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_LINUX

#endif // _LIBCPP_STACKTRACE_LINUX_H
