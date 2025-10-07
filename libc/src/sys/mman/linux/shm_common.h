//===---------- Shared implementations for shm_open/shm_unlink ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

// TODO: Get PATH_MAX via https://github.com/llvm/llvm-project/issues/85121
#include <linux/limits.h>

namespace LIBC_NAMESPACE_DECL {

namespace shm_common {

LIBC_INLINE_VAR constexpr cpp::string_view SHM_PREFIX = "/dev/shm/";
using SHMPath = cpp::array<char, NAME_MAX + SHM_PREFIX.size() + 1>;

LIBC_INLINE ErrorOr<SHMPath> translate_name(cpp::string_view name) {
  // trim leading slashes
  size_t offset = name.find_first_not_of('/');
  if (offset == cpp::string_view::npos)
    return Error(EINVAL);
  name = name.substr(offset);

  // check the name
  if (name.size() > NAME_MAX)
    return Error(ENAMETOOLONG);
  if (name == "." || name == ".." || name.contains('/'))
    return Error(EINVAL);

  // prepend the prefix
  SHMPath buffer;
  inline_memcpy(buffer.data(), SHM_PREFIX.data(), SHM_PREFIX.size());
  inline_memcpy(buffer.data() + SHM_PREFIX.size(), name.data(), name.size());
  buffer[SHM_PREFIX.size() + name.size()] = '\0';
  return buffer;
}
} // namespace shm_common

} // namespace LIBC_NAMESPACE_DECL
