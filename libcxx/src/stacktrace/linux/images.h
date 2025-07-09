//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_LINUX_IMAGES
#define _LIBCPP_STACKTRACE_LINUX_IMAGES

#if defined(__linux__)

#  include <__stacktrace/base.h>

#  include <array>
#  include <cassert>
#  include <cstddef>
#  include <cstdlib>
#  include <link.h>
#  include <unistd.h>

#  include "stacktrace/utils/image.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct images {
  // How many images this contains, including the left/right sentinels.
  unsigned count_{0};
  std::array<image, image::kMaxImages + 2> images_{};

  images();

  int add(dl_phdr_info& info);
  static int callback(dl_phdr_info* info, size_t, void* self);

  image& operator[](size_t index);

  image* mainProg();

  static images instance;
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // __linux__

#endif // _LIBCPP_STACKTRACE_LINUX_IMAGES
