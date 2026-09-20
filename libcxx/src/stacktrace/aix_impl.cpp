//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(_AIX)

#  include <__config>
#  include <__stacktrace/stacktrace_entry.h>
#  include <algorithm>
#  include <cerrno>
#  include <cstdlib>
#  include <mutex>
#  include <sys/ldr.h>
#  include <vector>

#  include "images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace __stacktrace {

namespace {

void scan_images(_Images& imgs) {
  std::vector<char> buf(512);
  while (loadquery(L_GETINFO, buf.data(), buf.size()) == -1) {
    if (errno == ENOMEM) {
      buf.resize(buf.size() * 2);
    } else {
      return;
    }
  }

  bool found_new      = false;
  struct ld_info* ldi = reinterpret_cast<struct ld_info*>(buf.data());
  while (imgs.count_ < _Images::k_max_images) {
    auto load_addr = reinterpret_cast<uintptr_t>(ldi->ldinfo_textorg);

    auto __end = imgs.images_.begin() + imgs.count_;
    auto __it  = std::lower_bound(imgs.images_.begin(), __end, load_addr, [](_Image const& __img, uintptr_t __addr) {
      return __img.load_addr_ < __addr;
    });
    if (__it == __end || __it->load_addr_ != load_addr) {
      auto is_first       = (imgs.count_ == 0);
      auto& image         = imgs.images_.at(imgs.count_++);
      image.load_addr_    = load_addr;
      image.slide_offset_ = load_addr;
      image.is_main_prog_ = is_first;

      const char* name = ldi->ldinfo_filename;
      // ldinfo_filename may not return the full path
      char resolved[_Entry::__max_file_len];
      if (name[0] != '/' && realpath(name, resolved) != nullptr) {
        name = resolved;
      }
      image.name_ = _Names::instance_.intern(name);
      found_new   = true;
    }

    if (ldi->ldinfo_next == 0) {
      break;
    }
    ldi = reinterpret_cast<struct ld_info*>(reinterpret_cast<char*>(ldi) + ldi->ldinfo_next);
  }

  if (found_new) {
    std::sort(imgs.images_.begin(), imgs.images_.begin() + imgs.count_);
  }
}

} // namespace

void _Images::enumerate() {
  std::lock_guard<std::mutex> __lock{mutex_};
  scan_images(*this);
}

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif
