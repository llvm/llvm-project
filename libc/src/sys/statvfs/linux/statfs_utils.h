//===-- Convert Statfs to Statvfs -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_STATVFS_LINUX_STATFS_TO_STATVFS_H
#define LLVM_LIBC_SRC_SYS_STATVFS_LINUX_STATFS_TO_STATVFS_H

#include "hdr/types/struct_statfs.h"
#include "include/llvm-libc-types/struct_statvfs.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace statfs_utils {

// Linux kernel set an additional flag to f_flags. Libc should mask it out.
LIBC_INLINE_VAR constexpr long ST_VALID = 0x0020;

// must use 'struct' tag to refer to type 'statvfs' in this scope. There will be
// a function in the same namespace with the same name. For consistency, we use
// struct prefix for all statvfs/statfs related types.
LIBC_INLINE struct statvfs statfs_to_statvfs(const struct statfs &in) {
  struct statvfs out{};
  out.f_bsize = in.f_bsize;
  out.f_frsize = in.f_frsize;
  out.f_blocks = static_cast<decltype(out.f_blocks)>(in.f_blocks);
  out.f_bfree = static_cast<decltype(out.f_bfree)>(in.f_bfree);
  out.f_bavail = static_cast<decltype(out.f_bavail)>(in.f_bavail);
  out.f_files = static_cast<decltype(out.f_files)>(in.f_files);
  out.f_ffree = static_cast<decltype(out.f_ffree)>(in.f_ffree);
  out.f_favail = static_cast<decltype(out.f_favail)>(in.f_ffree);
  out.f_fsid = in.f_fsid.__val[0];
  if constexpr (sizeof(decltype(out.f_fsid)) == sizeof(uint64_t))
    out.f_fsid |= static_cast<decltype(out.f_fsid)>(in.f_fsid.__val[1]) << 32;
  out.f_flag = in.f_flags & ~ST_VALID;
  out.f_namemax = in.f_namelen;
  return out;
}
} // namespace statfs_utils
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_STATVFS_LINUX_STATFS_TO_STATVFS_H
