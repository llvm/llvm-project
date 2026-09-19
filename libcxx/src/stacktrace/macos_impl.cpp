//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)

#  include <__stacktrace/stacktrace_entry.h>
#  include <algorithm>
#  include <cstdlib>
#  include <mutex>

#  include <dlfcn.h>
#  include <mach-o/dyld.h>
#  include <mach-o/loader.h>

#  include "images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace __stacktrace {

namespace {

// The callback we pass to dyld.  This synchronously receives each image at registration time,
// and asynchronously for each image loaded after initial registration.
void add_image(const struct mach_header* mh, intptr_t vmaddr_slide) {
  auto& imgs = _Images::instance_;
  std::lock_guard<std::mutex> __lock(imgs.mutex_);
  if (imgs.count_ == _Images::k_max_images) {
    return;
  }
  auto loaded_at = uintptr_t(mh);

  auto __end = imgs.images_.begin() + imgs.count_;
  auto __it  = std::lower_bound(imgs.images_.begin(), __end, loaded_at, [](_Image const& __img, uintptr_t __addr) {
    return __img.loaded_at_ < __addr;
  });
  if (__it != __end && __it->loaded_at_ == loaded_at) {
    return;
  }

  auto is_first       = (imgs.count_ == 0);
  auto& image         = imgs.images_.at(imgs.count_++);
  image.loaded_at_    = loaded_at;
  image.slide_        = uintptr_t(vmaddr_slide);
  image.is_main_prog_ = is_first;
  Dl_info __dl_info{};
  if (dladdr(mh, &__dl_info) && __dl_info.dli_fname) {
    image.name_ = _Names::instance_.intern(__dl_info.dli_fname);
  }
  std::sort(imgs.images_.begin(), imgs.images_.begin() + imgs.count_);
}

// Deferred to first use rather than done at library load, so a program that never calls
// std::stacktrace::current() never pays for walking the already-loaded image list. Must be
// called before `_Images::mutex()` is held: registration synchronously invokes `add_image`,
// which takes that same lock itself.
void ensure_registered() {
  static std::once_flag __once;
  std::call_once(__once, [] { _dyld_register_func_for_add_image(add_image); });
}

} // namespace

void _Images::refresh() { ensure_registered(); }

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS _LIBCPP_END_NAMESPACE_STD

#endif
