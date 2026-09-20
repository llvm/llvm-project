//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if !defined(_WIN32) && !defined(__APPLE__) && !defined(_AIX)
#  if __has_include("dlfcn.h") && __has_include("link.h")

#    include <__stacktrace/stacktrace_entry.h>
#    include <algorithm>
#    include <dlfcn.h>
#    include <link.h>
#    include <mutex>
#    include <unistd.h>

#    include "images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace {

using namespace __stacktrace;

struct _ScanState {
  _Images& images_;
  bool foundNew_{false};
};

int add_image(dl_phdr_info* info, size_t, void* state_v) {
  auto& state  = *static_cast<_ScanState*>(state_v);
  auto& images = state.images_;

  auto __end = images.images_.begin() + images.count_;
  auto __it =
      std::lower_bound(images.images_.begin(), __end, info->dlpi_addr, [](_Image const& __img, uintptr_t __addr) {
        return __img.load_addr_ < __addr;
      });
  if (__it != __end && __it->load_addr_ == info->dlpi_addr) {
    return 0;
  }

  if (images.count_ == _Images::__max_images) {
    return 1; // at capacity; stop iterating, nothing further can be added
  }
  auto is_first = (images.count_ == 0);
  auto& image   = images.images_.at(images.count_++);
  // Absolute address at which this ELF is loaded
  image.load_addr_ = info->dlpi_addr;
  // This also happens to be the "slide" amount since ELF has zero-relative offsets
  image.slide_offset_ = info->dlpi_addr;
  image.name_         = _Names::instance_.intern(info->dlpi_name);
  // `dl_iterate_phdr` gives us the main program image first
  image.is_main_prog_ = is_first;
  if (image.name_.empty() && is_first) {
    char buf[_Entry::__max_file_len]{0};
    if (readlink("/proc/self/exe", buf, sizeof(buf)) != -1) { // Ignores errno if error
      image.name_ = _Names::instance_.intern(buf);
    }
  }
  state.foundNew_ = true;
  return 0;
}

} // namespace

void __stacktrace::_Images::enumerate() {
  std::lock_guard<std::mutex> __lock(mutex_);
  _ScanState __state{*this};
  dl_iterate_phdr(add_image, &__state);
  if (__state.foundNew_) {
    std::sort(images_.begin(), images_.begin() + count_);
  }
}

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#  endif // __has_include("dlfcn.h") && __has_include("link.h")
#endif   // ! defined(_WIN32) && !defined(__APPLE__) && !defined(_AIX)
