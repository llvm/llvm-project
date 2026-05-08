//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(_AIX)

#  include <__config>
#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>
#  include <algorithm>
#  include <cstdlib>
#  include <sys/ldr.h>

#  include "stacktrace/images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

_Images::_Images() {
  std::vector<char> buf(512);
  while(loadquery(L_GETINFO, buf.data(), buf.size()) == -1) {
    if (errno == ENOMEM)
      buf.resize(buf.size() * 2);
    else
      return;
  }

  struct ld_info* ldi = reinterpret_cast<struct ld_info*>(buf.data());
  while (count_ < k_max_images - 2) {
    auto& image         = images_[count_++];
    image.slide_        = reinterpret_cast<uintptr_t>(ldi->ldinfo_textorg);
    image.loaded_at_    = image.slide_;
    image.is_main_prog_ = (count_ == 1);

    const char* name = ldi->ldinfo_filename;
    // ldinfo_filename may not return the full path
    if (name[0] != '/') {
      char resolved[_Entry::__max_file_len];
      if (realpath(name, resolved) != nullptr)
        name = resolved;
    }
    strncpy(image.name_, name, sizeof(image.name_));

    if (ldi->ldinfo_next == 0)
      break;
    
    ldi = reinterpret_cast<struct ld_info*>(reinterpret_cast<char*>(ldi) + ldi->ldinfo_next);
  }

  images_[count_++] = {0uz, 0};  // sentinel at low end
  images_[count_++] = {~0uz, 0}; // sentinel at high end

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
