//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// "Generic" implementation for any platform that doesn't have its own special implementation.
// (Currently this means any platform other than Windows)

#if !defined(_WIN32)

#  include <__config>
#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>

#  include "stacktrace/images.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

_LIBCPP_EXPORTED_FROM_ABI void base::find_images() {
  images images;
  size_t i = 0;
  for (auto& entry : __entry_iters_()) {
    images.find(&i, entry.__addr_);
    if (auto& image = images[i]) {
      entry.__image_ = &image;
      // While we're in this loop, get the executable's path, and tentatively use this for source file.
      entry.assign_file(__create_str()).assign(image.name_);
    }
  }
}

_LIBCPP_EXPORTED_FROM_ABI void base::find_symbols() {
  // TODO
}

_LIBCPP_EXPORTED_FROM_ABI void base::find_source_locs() {
  // TODO
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
