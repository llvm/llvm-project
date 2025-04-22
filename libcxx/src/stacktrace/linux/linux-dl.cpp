//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/config.h"

#if defined(_LIBCPP_STACKTRACE_LINUX)

#  include "linux.h"

#  include <cassert>
#  include <dlfcn.h>
#  include <link.h>
#  include <unistd.h>

#  include "../common/images.h"

#  include <__stacktrace/context.h>
#  include <__stacktrace/entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

void linux::ident_modules() {
  auto& images = images::get();

  // Aside from the left/right sentinels in the array (hence the 2),
  // are there any other real images?
  if (images.count_ <= 2) {
    return;
  }

  auto mainProg = images.mainProg();
  if (mainProg) {
    cx_.__main_prog_path_ = mainProg->name_;
  }

  unsigned index = 1; // Starts at one, and is moved around in this loop
  for (auto& entry : cx_.__entries_) {
    while (images[index].loaded_at_ > entry.__addr_) {
      --index;
    }
    while (images[index + 1].loaded_at_ <= entry.__addr_) {
      ++index;
    }
    entry.__addr_unslid_ = entry.__addr_ - images[index].slide_;
    entry.__file_        = images[index].name_;
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_LINUX
