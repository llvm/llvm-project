//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of backtrace_symbols
///
//===----------------------------------------------------------------------===//

#include "src/execinfo/backtrace_symbols.h"

#include "hdr/func/malloc.h"
#include "hdr/types/dl_info.h"
#include "hdr/types/size_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dladdr.h"
#include "src/execinfo/backtrace_line.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

// Renders one line into `out`, which has `room` bytes, and returns how many
// bytes it took including the terminator. A null `out` measures instead.
size_t render(void *address, char *out, size_t room) {
  char scratch[execinfo::MAX_LINE];
  char *cursor = scratch;
  char *const end = scratch + sizeof(scratch) - 1;

  Dl_info info = {};
  const bool found = internal::dladdr_impl(address, &info) != 0;
  if (found && info.dli_fname != nullptr)
    cursor = execinfo::put(cursor, end, info.dli_fname);
  if (found && info.dli_sname != nullptr) {
    cursor = execinfo::put(cursor, end, '(');
    cursor = execinfo::put(cursor, end, info.dli_sname);
    const uintptr_t offset = reinterpret_cast<uintptr_t>(address) -
                             reinterpret_cast<uintptr_t>(info.dli_saddr);
    cursor = execinfo::put(cursor, end, '+');
    cursor = execinfo::put_hex(cursor, end, offset);
    cursor = execinfo::put(cursor, end, ')');
  }
  if (cursor != scratch)
    cursor = execinfo::put(cursor, end, ' ');
  cursor = execinfo::put(cursor, end, '[');
  cursor = execinfo::put_hex(cursor, end, reinterpret_cast<uintptr_t>(address));
  cursor = execinfo::put(cursor, end, ']');
  *cursor = '\0';

  const size_t length = static_cast<size_t>(cursor - scratch) + 1;
  if (out != nullptr && room >= length)
    for (size_t i = 0; i < length; ++i)
      out[i] = scratch[i];
  return length;
}

} // anonymous namespace

// The same lines backtrace_symbols_fd writes, in one block the caller frees
// with a single call: the array of pointers is at the front and the strings
// follow it.
LLVM_LIBC_FUNCTION(char **, backtrace_symbols,
                   (void *const *buffer, int size)) {
  if (buffer == nullptr || size <= 0)
    return nullptr;

  const size_t count = static_cast<size_t>(size);
  size_t total = count * sizeof(char *);
  for (size_t i = 0; i < count; ++i)
    total += render(buffer[i], nullptr, 0);

  char **result = static_cast<char **>(malloc(total));
  if (result == nullptr)
    return nullptr;

  char *text = reinterpret_cast<char *>(result) + count * sizeof(char *);
  size_t left = total - count * sizeof(char *);
  for (size_t i = 0; i < count; ++i) {
    result[i] = text;
    const size_t used = render(buffer[i], text, left);
    text += used;
    left -= used;
  }
  return result;
}

} // namespace LIBC_NAMESPACE_DECL
