//===-- Simple malloc and free for use with integration tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>
#include <stdint.h>

// Integration tests cannot use the SCUDO standalone allocator as SCUDO pulls
// various other parts of the libc. Since SCUDO development does not use
// LLVM libc build rules, it is very hard to keep track or pull all that SCUDO
// requires. Hence, as a work around for this problem, we use a simple allocator
// which just hands out continuous blocks from a statically allocated chunk of
// memory.

static uint8_t memory[16384];
static uint8_t *ptr = memory;

extern "C" {

void *malloc(size_t s) {
  void *mem = ptr;
  ptr += s;
  return mem;
}

void free(void *) {}

} // extern "C"
