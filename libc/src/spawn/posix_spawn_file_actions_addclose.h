//===-- Impl header for posix_spawn_file_actions_addclose -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SPAWN_POSIX_SPAWN_FILE_ACTIONS_ADDCLOSE_H
#define LLVM_LIBC_SRC_SPAWN_POSIX_SPAWN_FILE_ACTIONS_ADDCLOSE_H

#include <spawn.h>

namespace __llvm_libc {

int posix_spawn_file_actions_addclose(
    posix_spawn_file_actions_t *__restrict actions, int fd);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SPAWN_POSIX_SPAWN_FILE_ACTIONS_ADDCLOSE_H
