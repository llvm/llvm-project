//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)

#  include <__config>
#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>
#  include <algorithm>
#  include <cstdlib>
#  include <dlfcn.h>
#  include <mach-o/dyld.h>
#  include <mach-o/loader.h>

#  include "stacktrace/images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

_Images::_Images() {
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

_LIBCPP_EXPORTED_FROM_ABI void _Trace::populate_images() {
  _Images images;
  size_t i = 0;
  for (auto& entry : __entry_iters_()) {
    images.find(&i, entry.__addr_);
    if (auto& image = images[i]) {
      entry.__image_ = &image;
      // While we're in this loop, get the executable's path, and tentatively use this for source file.
      entry.__file_.assign(image.name_);
    }
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
