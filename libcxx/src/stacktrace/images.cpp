//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if !defined(_WIN32) // Separate impl exists for windows

#  include <__config>
#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>
#  include <mutex>

#  include "images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace __stacktrace {

_Names _Names::instance_;
_Images _Images::instance_;

// See the comment on the declaration in images.h: this weak no-op is only what actually links
// when no platform impl file provides a strong override.
__attribute__((__weak__)) void _Images::enumerate() {}

void __populate_images(_Context& __cx) {
  _Images& images = _Images::instance_;
  images.enumerate();

  std::lock_guard<std::mutex> __lock(images.mutex_);
  for (auto& entry : __cx.__entry_iters_(__cx.__self_)) {
    auto __i = images.find(entry.__addr_);
    if (auto& image = images[__i]) {
      entry.__image_ = &image;
      entry.__file_  = image.name_; // tentatively used as source filename unless lookup succeeds later
    }
  }
}

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_WIN32)
