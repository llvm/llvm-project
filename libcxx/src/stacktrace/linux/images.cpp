//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__linux__)

#  include <__stacktrace/base.h>

#  include "stacktrace/linux/images.h"
#  include "stacktrace/utils/image.h"

#  include <algorithm>
#  include <cassert>
#  include <cstddef>
#  include <cstdlib>
#  include <link.h>
#  include <unistd.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

images images::instance;

int images::add(dl_phdr_info& info) {
  assert(count_ < image::kMaxImages);
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
  return count_ == image::kMaxImages; // return nonzero if we're at the limit
}

int images::callback(dl_phdr_info* info, size_t, void* self) { return (*(images*)(self)).add(*info); }

images::images() {
  dl_iterate_phdr(images::callback, this);
  images_[count_++] = {0uz, 0};  // sentinel at low end
  images_[count_++] = {~0uz, 0}; // sentinel at high end
  std::sort(images_.begin(), images_.begin() + count_ - 1);
}

image& images::operator[](size_t index) {
  assert(index < count_);
  return images_.at(index);
}

image* images::mainProg() {
  for (auto& image : images_) {
    if (image.is_main_prog_) {
      return &image;
    }
  }
  return nullptr;
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // __linux__
