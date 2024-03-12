//===---------- Shared implementations for shm_open/shm_unlink ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/string_view.h"
#include "src/errno/libc_errno.h"
#include "src/string/memchr.h"
#include "src/string/memcpy.h"
#include <asm/errno.h>
#include <linux/limits.h>

namespace LIBC_NAMESPACE {

LIBC_INLINE_VAR constexpr cpp::string_view SHM_PREFIX = "/dev/shm/";
using SHMPath = cpp::array<char, NAME_MAX + SHM_PREFIX.size() + 1>;

LIBC_INLINE cpp::optional<SHMPath> get_shm_name(cpp::string_view name) {
  // trim leading slashes
  size_t offset = name.find_first_not_of('/');
  if (offset == cpp::string_view::npos) {
    libc_errno = EINVAL;
    return cpp::nullopt;
  }
  name = name.substr(offset);

  // check the name
  if (name.size() > NAME_MAX) {
    libc_errno = ENAMETOOLONG;
    return cpp::nullopt;
  }
  if (name == "." || name == ".." ||
      memchr(name.data(), '/', name.size()) != nullptr) {
    libc_errno = EINVAL;
    return cpp::nullopt;
  }

  // prepend the prefix
  SHMPath buffer;
  memcpy(buffer.data(), SHM_PREFIX.data(), SHM_PREFIX.size());
  memcpy(buffer.data() + SHM_PREFIX.size(), name.data(), name.size());
  buffer[SHM_PREFIX.size() + name.size()] = '\0';
  return buffer;
}

} // namespace LIBC_NAMESPACE
