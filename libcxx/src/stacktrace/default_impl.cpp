//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if !defined(_WIN32) && !defined(__APPLE__) && !defined(_AIX)

#  include <__config>
#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>
#  include <algorithm>
#  include <unistd.h>

#  if __has_include("dlfcn.h")
#    include <dlfcn.h>
#  endif
#  if __has_include("link.h")
#    include <link.h>
#  endif

#  include "stacktrace/images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace __stacktrace {

namespace {

#  if __has_include("dlfcn.h") && __has_include("link.h")
int add_image(dl_phdr_info* info, size_t, void* images_v) {
  auto& imgs = *(_Images*)images_v;
  if (imgs.count_ == _Images::k_max_images) {
    return 0;
  }
  auto is_first = (imgs.count_ == 0);
  auto& image   = imgs.images_.at(imgs.count_++);
  // Absolute address at which this ELF is loaded
  image.loaded_at_ = info->dlpi_addr;
  // This also happens to be the "slide" amount since ELF has zero-relative offsets
  image.slide_ = info->dlpi_addr;
  strncpy(image.name_, info->dlpi_name, sizeof(image.name_));
  // `dl_iterate_phdr` gives us the main program image first
  image.is_main_prog_ = is_first;
  if (!image.name_[0] && is_first) {
    char buf[_Entry::__max_file_len]{0};
    if (readlink("/proc/self/exe", buf, sizeof(buf)) != -1) { // Ignores errno if error
      strncpy(image.name_, buf, sizeof(image.name_));
    }
  }
  // If we're at the limit, return nonzero to stop iterating
  return imgs.count_ == _Images::k_max_images;
}
#  endif

} // namespace

_Images::_Images() {
#  if __has_include("dlfcn.h") && __has_include("link.h")
  dl_iterate_phdr(add_image, this);
#  endif
  images_[count_++] = {0uz, 0};  // sentinel at low end
  images_[count_++] = {~0uz, 0}; // sentinel at high end
  std::sort(images_.begin(), images_.begin() + count_);
}

void _Trace::__populate_images() {
  _Images images;
  size_t i = 0;
  for (auto& entry : __entry_iters_()) {
    images.find(&i, entry.__addr_);
    if (auto& image = images[i]) {
      entry.__image_ = &image;
      // While we're in this loop, get the executable's path, and tentatively use this for source file.
      entry.__file_.__assign(image.name_);
    }
  }
}

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif
