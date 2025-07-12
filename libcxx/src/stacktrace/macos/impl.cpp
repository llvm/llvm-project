//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)

#  include <algorithm>
#  include <array>
#  include <dlfcn.h>
#  include <mach-o/dyld.h>
#  include <mach-o/loader.h>

#  include <__stacktrace/base.h>

#  include "stacktrace/macos/impl.h"
#  include "stacktrace/utils/image.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

void ident_module(alloc& alloc, entry_base& entry, unsigned& index, image* images) {
  if (entry.__addr_actual_) {
    while (images[index].loaded_at_ > entry.__addr_actual_) {
      --index;
    }
    while (images[index + 1].loaded_at_ <= entry.__addr_actual_) {
      ++index;
    }
    auto& image = images[index];
    if (image) {
      entry.__addr_unslid_ = entry.__addr_actual_ - images[index].slide_;
      entry.__file_        = alloc.make_str(images[index].name_);
    }
  }
}

bool enum_modules(unsigned& count, auto& images) {
  count = std::min(image::kMaxImages, size_t(_dyld_image_count()));
  for (size_t i = 0; i < count; i++) {
    auto& image      = images[i];
    image.slide_     = _dyld_get_image_vmaddr_slide(i);
    image.loaded_at_ = uintptr_t(_dyld_get_image_header(i));
    image.name_      = _dyld_get_image_name(i);
  }
  images[count++] = {0uz, 0};  // sentinel at low end
  images[count++] = {~0uz, 0}; // sentinel at high end
  std::sort(images.begin(), images.begin() + count - 1);
  return true;
}

void macos::ident_modules() {
  static unsigned imageCount;
  static std::array<image, image::kMaxImages + 2> images;
  static bool atomicInitialized = enum_modules(imageCount, images);
  (void)atomicInitialized;

  // Aside from the left/right sentinels in the array (hence the 2),
  // are there any other real images?
  if (imageCount <= 2) {
    return;
  }

  // First image (the main program) is at index 1
  builder_.__main_prog_path_ = builder_.__alloc_.make_str(images.at(1).name_);

  unsigned index = 1; // Starts at one, and is moved by 'ident_module'
  for (auto& entry : builder_.__entries_) {
    ident_module(builder_.__alloc_, (entry_base&)entry, index, images.data());
  }
}

void symbolize_entry(alloc& alloc, entry_base& entry) {
  Dl_info info;
  if (dladdr((void*)entry.__addr_actual_, &info)) {
    if (info.dli_fname && entry.__file_->empty()) {
      // provide at least the binary filename in case we cannot lookup source location
      entry.__file_ = alloc.make_str(info.dli_fname);
    }
    if (info.dli_sname && entry.__desc_->empty()) {
      // provide at least the mangled name; try to unmangle in a later step
      entry.__desc_ = alloc.make_str(info.dli_sname);
    }
  }
}

void macos::symbolize() {
  for (auto& entry : builder_.__entries_) {
    symbolize_entry(builder_.__alloc_, (entry_base&)entry);
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // __APPLE__
