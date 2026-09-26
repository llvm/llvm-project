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
  auto& images = _Images::instance_;
  std::lock_guard<std::mutex> __lock(images.mutex_);

  if (images.count_ == _Images::__max_images) {
    return;
  }
  auto load_addr = uintptr_t(mh);

  auto __end = images.images_.begin() + images.count_;
  auto __it  = std::lower_bound(images.images_.begin(), __end, load_addr, [](_Image const& __img, uintptr_t __addr) {
    return __img.load_addr_ < __addr;
  });
  if (__it != __end && __it->load_addr_ == load_addr) {
    return;
  }

  auto& image         = images.images_.at(images.count_++);
  image.load_addr_    = load_addr;
  image.slide_offset_ = uintptr_t(vmaddr_slide);
  image.is_main_prog_ = (images.count_ == 0);
  Dl_info __dl_info{};
  if (dladdr(mh, &__dl_info) && __dl_info.dli_fname) {
    image.name_ = _Names::instance_.intern(__dl_info.dli_fname);
  }
  std::sort(images.images_.begin(), images.images_.begin() + images.count_);
}

// Note: `_Images::instance_.mutex_` must not be held
void ensure_registered() {
  static std::once_flag __once;
  std::call_once(__once, [] { _dyld_register_func_for_add_image(add_image); });
}

} // namespace

void _Images::enumerate() { ensure_registered(); }

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS _LIBCPP_END_NAMESPACE_STD

#endif
