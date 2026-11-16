//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getdelim logic shared by getline and getdelim.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_GENERIC_GETDELIM_IMPL_H
#define LLVM_LIBC_SRC_STDIO_GENERIC_GETDELIM_IMPL_H

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/unique_ptr.h"
#include "src/__support/File/file.h"
#include "src/__support/checked_ptr.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

LIBC_INLINE constexpr size_t default_max_capacity() {
  if constexpr (cpp::numeric_limits<ssize_t>::digits <
                cpp::numeric_limits<size_t>::digits) {
    return static_cast<size_t>(cpp::numeric_limits<ssize_t>::max()) + 1;
  } else {
    return cpp::numeric_limits<size_t>::max();
  }
}

// Manages the caller-owned (*lineptr, *n) buffer via RAII so that any newly
// allocated or grown buffer and its capacity are always written back to the
// caller's pointers on scope exit, keeping *lineptr valid for free() even if
// reading or subsequent growth fails.
// TODO: Replace with a dynamic array container once available.
template <size_t MaxCapacity> class LineBuffer {
  static_assert(MaxCapacity >= 2, "MaxCapacity must be at least 2");

  static constexpr size_t INITIAL_CAPACITY = 64;

  char *&out_ptr;
  size_t &out_cap;
  cpp::unique_ptr<char[]> buffer;
  size_t length = 0;
  size_t capacity = 0;

  LIBC_INLINE bool ensure_capacity() {
    // Ensure `length + 2` (for the incoming char and trailing null terminator)
    // cannot overflow `size_t` or exceed `MaxCapacity` (which is >= 2).
    if (LIBC_UNLIKELY(length > MaxCapacity - 2))
      return false;

    if (LIBC_LIKELY(capacity >= length + 2))
      return true;

    size_t new_capacity = next_capacity(capacity);

    CheckedPtr<char[]> bigger_buffer =
        make_unique_for_overwrite<char[]>(new_capacity);
    if (LIBC_UNLIKELY(!bigger_buffer))
      return false;

    if (length > 0)
      inline_memcpy(bigger_buffer.get(), buffer.get(), length);

    capacity = new_capacity;
    buffer = bigger_buffer.take_unique();
    return true;
  }

public:
  LIBC_INLINE static constexpr size_t next_capacity(size_t current_capacity) {
    if constexpr (MaxCapacity <= INITIAL_CAPACITY) {
      return MaxCapacity;
    } else {
      if (current_capacity < INITIAL_CAPACITY / 2)
        return INITIAL_CAPACITY;
      if (current_capacity > MaxCapacity / 2)
        return MaxCapacity;
      return current_capacity * 2;
    }
  }

  LIBC_INLINE LineBuffer(char *&ptr, size_t &cap)
      : out_ptr(ptr), out_cap(cap), buffer(ptr),
        capacity(ptr != nullptr ? cap : 0) {
    out_ptr = nullptr;
  }

  LineBuffer(const LineBuffer &) = delete;
  LineBuffer &operator=(const LineBuffer &) = delete;
  LineBuffer(LineBuffer &&) = delete;
  LineBuffer &operator=(LineBuffer &&) = delete;

  LIBC_INLINE ~LineBuffer() {
    if (buffer) {
      out_cap = capacity;
      out_ptr = buffer.release();
    }
  }

  LIBC_INLINE constexpr size_t size() const { return length; }

  [[nodiscard]] LIBC_INLINE bool push_back(char c) {
    if (LIBC_UNLIKELY(!ensure_capacity()))
      return false;

    buffer[length++] = c;
    return true;
  }

  LIBC_INLINE void null_terminate() { buffer[length] = '\0'; }
};

template <size_t MaxCapacity = default_max_capacity()>
LIBC_INLINE ErrorOr<ssize_t> getdelim_impl(char **__restrict lineptr,
                                           size_t *__restrict n, int delim,
                                           File *__restrict stream) {
  static_assert(MaxCapacity >= 2 && MaxCapacity <= default_max_capacity(),
                "MaxCapacity must be within [2, default_max_capacity()]");
  File::FileLock lock(stream);

  if (lineptr == nullptr || n == nullptr) {
    stream->set_error_unlocked();
    return Error(EINVAL);
  }

  if (stream->try_set_orientation_unlocked(File::Orientation::BYTE) !=
      File::Orientation::BYTE) {
    stream->set_error_unlocked();
    return Error(EINVAL);
  }

  if (stream->iseof_unlocked())
    return -1;

  LineBuffer<MaxCapacity> line_buf(*lineptr, *n);
  const unsigned char target = static_cast<unsigned char>(delim);

  while (true) {
    unsigned char c = 0;
    auto result = stream->read_unlocked(&c, 1);
    if (LIBC_UNLIKELY(result.has_error())) {
      stream->set_error_unlocked();
      return Error(result.error);
    }
    if (result.value != 1)
      break;

    if (LIBC_UNLIKELY(line_buf.size() == MaxCapacity - 1)) {
      stream->set_error_unlocked();
      return Error(EOVERFLOW);
    }

    if (LIBC_UNLIKELY(!line_buf.push_back(static_cast<char>(c)))) {
      stream->set_error_unlocked();
      return Error(ENOMEM);
    }

    if (c == target)
      break;
  }

  if (line_buf.size() == 0)
    return -1;

  line_buf.null_terminate();
  return static_cast<ssize_t>(line_buf.size());
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_GENERIC_GETDELIM_IMPL_H
