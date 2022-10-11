//===-- Impl header for posix_spawn_file_actions_adddup2 --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SPAWN_POSIX_SPAWN_FILE_ACTIONS_ADDDUP2_H
#define LLVM_LIBC_SRC_SPAWN_POSIX_SPAWN_FILE_ACTIONS_ADDDUP2_H

#include <spawn.h>

namespace __llvm_libc {

int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t *actions,
                                     int fd, int newfd);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SPAWN_POSIX_SPAWN_FILE_ACTIONS_ADDDUP2_H
