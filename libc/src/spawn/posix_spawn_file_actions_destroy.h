//===-- Impl header for posix_spawn_file_actions_destroy --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SPAWN_POSIX_SPAWN_FILE_ACTIONS_DESTROY_H
#define LLVM_LIBC_SRC_SPAWN_POSIX_SPAWN_FILE_ACTIONS_DESTROY_H

#include <spawn.h>

namespace __llvm_libc {

int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t *actions);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SPAWN_POSIX_SPAWN_FILE_ACTIONS_DESTROY_H
