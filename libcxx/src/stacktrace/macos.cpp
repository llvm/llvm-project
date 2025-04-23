//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "stacktrace/config.h"

#if defined(_LIBCPP_STACKTRACE_MACOS)

#  include "stacktrace/context.h"
#  include "stacktrace/macos.h"

#  include <algorithm>
#  include <array>
#  include <dlfcn.h>
#  include <mach-o/dyld.h>
#  include <mach-o/loader.h>

#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

constexpr unsigned kMaxImages = 256;

struct Image {
  uintptr_t loadedAt{};
  intptr_t slide{};
  std::string_view name{};
  bool operator<(Image const& rhs) const { return loadedAt < rhs.loadedAt; }
  operator bool() const { return !name.empty(); }
};

void ident_module(entry& entry, unsigned& index, Image* images) {
  if (entry.__addr_) {
    while (images[index].loadedAt > entry.__addr_) {
      --index;
    }
    while (images[index + 1].loadedAt <= entry.__addr_) {
      ++index;
    }
    auto& image = images[index];
    if (image) {
      entry.__addr_unslid_ = entry.__addr_ - images[index].slide;
      entry.__file_        = images[index].name;
    }
  }
}

bool enum_modules(unsigned& count, auto& images) {
  count = std::min(kMaxImages, _dyld_image_count());
  for (size_t i = 0; i < count; i++) {
    auto& image    = images[i];
    image.slide    = _dyld_get_image_vmaddr_slide(i);
    image.loadedAt = uintptr_t(_dyld_get_image_header(i));
    image.name     = _dyld_get_image_name(i);
  }
  images[count++] = {0uz, 0};  // sentinel at low end
  images[count++] = {~0uz, 0}; // sentinel at high end
  std::sort(images.begin(), images.begin() + count - 1);
  return true;
}

void macos::ident_modules() {
  static unsigned imageCount;
  static std::array<Image, kMaxImages + 2> images;
  static bool atomicInitialized = enum_modules(imageCount, images);
  (void)atomicInitialized;

  // Aside from the left/right sentinels in the array (hence the 2),
  // are there any other real images?
  if (imageCount <= 2) {
    return;
  }

  // First image (the main program) is at index 1
  cx_.__main_prog_path_ = images.at(1).name;

  unsigned index = 1; // Starts at one, and is moved by 'ident_module'
  for (auto& entry : cx_.__entries_) {
    ident_module(entry, index, images.data());
  }
}

void symbolize_entry(entry& entry) {
  Dl_info info;
  if (dladdr((void*)entry.__addr_, &info)) {
    if (info.dli_fname && entry.__file_.empty()) {
      // provide at least the binary filename in case we cannot lookup source location
      entry.__file_ = info.dli_fname;
    }
    if (info.dli_sname && entry.__desc_.empty()) {
      // provide at least the mangled name; try to unmangle in a later step
      entry.__desc_ = info.dli_sname;
    }
  }
}

void macos::symbolize() {
  for (auto& entry : cx_.__entries_) {
    symbolize_entry(entry);
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_MACOS
