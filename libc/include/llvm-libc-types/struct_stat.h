//===-- Definition of struct stat -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_STAT_H
#define LLVM_LIBC_TYPES_STRUCT_STAT_H

#include <llvm-libc-types/blkcnt_t.h>
#include <llvm-libc-types/blksize_t.h>
#include <llvm-libc-types/dev_t.h>
#include <llvm-libc-types/gid_t.h>
#include <llvm-libc-types/ino_t.h>
#include <llvm-libc-types/mode_t.h>
#include <llvm-libc-types/nlink_t.h>
#include <llvm-libc-types/off_t.h>
#include <llvm-libc-types/struct_timespec.h>
#include <llvm-libc-types/uid_t.h>

struct stat {
  dev_t st_dev;
  ino_t st_ino;
  mode_t st_mode;
  nlink_t st_nlink;
  uid_t st_uid;
  gid_t st_gid;
  dev_t st_rdev;
  off_t st_size;
  struct timespec st_atim;
  struct timespec st_mtim;
  struct timespec st_ctim;
  blksize_t st_blksize;
  blkcnt_t st_blocks;
};

#endif // LLVM_LIBC_TYPES_STRUCT_STAT_H
