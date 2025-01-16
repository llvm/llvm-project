//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// OS-specific construction
//

#include "__config"

#if defined(__APPLE__)
// MacOS-specific: use the `dyld` loader to access info about loaded Mach-O images.
#  include "stacktrace/images.h"
#  include <algorithm>
#  include <cstdlib>
#  include <dlfcn.h>
#  include <mach-o/dyld.h>
#  include <mach-o/loader.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

images::images() {
  images_[count_++] = {0uz, 0};  // sentinel at low end
  images_[count_++] = {~0uz, 0}; // sentinel at high end
  auto dyld_count   = _dyld_image_count();
  for (unsigned i = 0; i < dyld_count && count_ < k_max_images; i++) {
    auto& image         = images_[count_++];
    image.slide_        = uintptr_t(_dyld_get_image_vmaddr_slide(i));
    image.loaded_at_    = uintptr_t(_dyld_get_image_header(i));
    image.is_main_prog_ = (i == 0);
    strncpy(image.name_, _dyld_get_image_name(i), sizeof(image.name_));
  }
  std::sort(images_.begin(), images_.begin() + count_);
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#elif !defined(_WIN32)
// Non-MacOS and non-Windows, including Linux: assume environment has these headers.
#  include "stacktrace/images.h"
#  include <algorithm>
#  include <cstdlib>
#  include <dlfcn.h>
#  include <link.h>
#  include <unistd.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

namespace {
int add_image(dl_phdr_info* info, size_t, void* images_v) {
  auto& imgs = *(images*)images_v;
  if (imgs.count_ == images::k_max_images) {
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
    char buf[entry_base::__max_file_len]{0};
    if (readlink("/proc/self/exe", buf, sizeof(buf)) != -1) { // Ignores errno if error
      strncpy(image.name_, buf, sizeof(image.name_));
    }
  }
  // If we're at the limit, return nonzero to stop iterating
  return imgs.count_ == images::k_max_images;
}
} // namespace

images::images() {
  dl_iterate_phdr(add_image, this);
  images_[count_++] = {0uz, 0};  // sentinel at low end
  images_[count_++] = {~0uz, 0}; // sentinel at high end
  std::sort(images_.begin(), images_.begin() + count_);
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
