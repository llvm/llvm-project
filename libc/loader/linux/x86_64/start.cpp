//===------------------ Implementation of crt for x86_64 ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/syscall.h"
#include "include/sys/syscall.h"

#include <linux/auxvec.h>
#include <stdint.h>

extern "C" int main(int, char **, char **);

struct Args {
  // At the language level, argc is an int. But we use uint64_t as the x86_64
  // ABI specifies it as an 8 byte value.
  uint64_t argc;

  // At the language level, argv is a char** value. However, we use uint64_t as
  // the x86_64 ABI specifies the argv vector be an |argc| long array of 8-byte
  // values. Even though a flexible length array would be more suitable here, we
  // set the array length to 1 to avoid a compiler warning about it being a C99
  // extension. Length of 1 is not really wrong as |argc| is guaranteed to be
  // atleast 1, and there is an 8-byte null entry at the end of the argv array.
  uint64_t argv[1];
};

// TODO: Would be nice to use the aux entry structure from elf.h when available.
struct AuxEntry {
  uint64_t type;
  uint64_t value;
};

extern "C" void _start() {
  uintptr_t *frame_ptr =
      reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));

  // This TU is compiled with -fno-omit-frame-pointer. Hence, the previous value
  // of the base pointer is pushed on to the stack. So, we step over it (the
  // "+ 1" below) to get to the args.
  Args *args = reinterpret_cast<Args *>(frame_ptr + 1);

  // After the argv array, is a 8-byte long NULL value before the array of env
  // values. The end of the env values is marked by another 8-byte long NULL
  // value. We step over it (the "+ 1" below) to get to the env values.
  uint64_t *env_ptr = args->argv + args->argc + 1;
  uint64_t *env_end_marker = env_ptr;
  while (*env_end_marker)
    ++env_end_marker;

  // After the env array, is the aux-vector. The end of the aux-vector is
  // denoted by an AT_NULL entry.
  for (AuxEntry *aux_entry = reinterpret_cast<AuxEntry *>(env_end_marker + 1);
       aux_entry->type != AT_NULL; ++aux_entry) {
    // TODO: Read the aux vector and store necessary information in a libc wide
    // data structure.
  }

  __llvm_libc::syscall(SYS_exit,
                       main(args->argc, reinterpret_cast<char **>(args->argv),
                            reinterpret_cast<char **>(env_ptr)));
}
