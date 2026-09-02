//===-- Writer definition for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_WRITER_H
#define LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_WRITER_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/printf_core/core_structs.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {

#define HANDLE_OVERFLOW_MODE(MODE) MODE,
enum class OverflowMode {
#include "src/__support/printf_core/overflow_modes.def"
};
#undef HANDLE_OVERFLOW_MODE

// Helper to omit the template argument if we are using runtime dispatch and
// avoid multiple copies of the converter functions.
template <OverflowMode mode> struct Mode {
#ifdef LIBC_COPT_PRINTF_RUNTIME_DISPATCH
  static constexpr OverflowMode value = OverflowMode::RUNTIME_DISPATCH;
#else
  static constexpr OverflowMode value = mode;
#endif
};

template <typename CharT> struct WriteBuffer {
  CharT *buff;
  size_t buff_len;
  size_t buff_cur = 0;
};

// Function type for handling the slow path in `Writer` when the `WriteBuffer`
// may not have enough remaining capacity for the new content.
template <typename CharT>
using OverflowWriteFn = int (*)(WriteBuffer<CharT> & /* wb */,
                                cpp::basic_string_view<CharT> /* new_str */,
                                void * /* state */);

// Handles overflow by filling any remaining space in `wb` with the start of
// `new_str`, and dropping any excess bytes.
template <typename CharT>
LIBC_INLINE int
overflow_write_drop_overflow(WriteBuffer<CharT> &wb,
                             cpp::basic_string_view<CharT> new_str, void *) {
  if (wb.buff_cur < wb.buff_len) {
    size_t chars_to_write = wb.buff_len - wb.buff_cur;
    if (chars_to_write > new_str.size())
      chars_to_write = new_str.size();
    inline_memcpy(wb.buff + wb.buff_cur, new_str.data(),
                  chars_to_write * sizeof(CharT));
    wb.buff_cur += chars_to_write;
  }
  return WRITE_OK;
}

// Defined in make_file_writer.h
template <typename CharT>
int overflow_write_flush_to_file(WriteBuffer<CharT> &wb,
                                 cpp::basic_string_view<CharT> new_str,
                                 void *fp);

// Defined in make_stderr_writer.h
template <typename CharT>
int overflow_write_flush_to_stderr(WriteBuffer<CharT> &wb,
                                   cpp::basic_string_view<CharT> new_str,
                                   void *);

// Defined in make_resizing_writer.h
template <typename CharT>
int overflow_write_resize_buffer(WriteBuffer<CharT> &wb,
                                 cpp::basic_string_view<CharT> new_str,
                                 void *initial_stack_buffer);

// Trait type helper mapping `OverflowMode` to the corresponding function.
template <OverflowMode mode, typename CharT> struct OverflowModeToWriteFn {};

template <typename CharT>
struct OverflowModeToWriteFn<OverflowMode::DROP_OVERFLOW, CharT> {
  static constexpr OverflowWriteFn<CharT> value =
      overflow_write_drop_overflow<CharT>;
};

template <typename CharT>
struct OverflowModeToWriteFn<OverflowMode::FLUSH_TO_FILE, CharT> {
  static constexpr OverflowWriteFn<CharT> value =
      overflow_write_flush_to_file<CharT>;
};

template <typename CharT>
struct OverflowModeToWriteFn<OverflowMode::FLUSH_TO_STDERR, CharT> {
  static constexpr OverflowWriteFn<CharT> value =
      overflow_write_flush_to_stderr<CharT>;
};

template <typename CharT>
struct OverflowModeToWriteFn<OverflowMode::RESIZE_BUFFER, CharT> {
  static constexpr OverflowWriteFn<CharT> value =
      overflow_write_resize_buffer<CharT>;
};

// Helper template used by `Writer` to dispatch to the appropriate
// `OverflowWriteFn`.
template <OverflowMode mode, typename CharT> struct OverflowWriter;

template <typename CharT>
struct OverflowWriter<OverflowMode::DROP_OVERFLOW, CharT> {
  LIBC_INLINE OverflowWriter(OverflowWriteFn<CharT>, void *) {}

  LIBC_INLINE int write(WriteBuffer<CharT> &wb,
                        cpp::basic_string_view<CharT> new_str) {
    return overflow_write_drop_overflow<CharT>(wb, new_str, nullptr);
  }
};

template <typename CharT>
struct OverflowWriter<OverflowMode::FLUSH_TO_FILE, CharT> {
  void *file;

  LIBC_INLINE OverflowWriter(OverflowWriteFn<CharT>, void *file) : file(file) {}

  LIBC_INLINE int write(WriteBuffer<CharT> &wb,
                        cpp::basic_string_view<CharT> new_str) {
    return overflow_write_flush_to_file(wb, new_str, file);
  }
};

template <typename CharT>
struct OverflowWriter<OverflowMode::FLUSH_TO_STDERR, CharT> {
  LIBC_INLINE OverflowWriter(OverflowWriteFn<CharT>, void *) {}

  LIBC_INLINE int write(WriteBuffer<CharT> &wb,
                        cpp::basic_string_view<CharT> new_str) {
    return overflow_write_flush_to_stderr(wb, new_str, nullptr);
  }
};

template <typename CharT>
struct OverflowWriter<OverflowMode::RESIZE_BUFFER, CharT> {
  void *initial_stack_buffer;

  LIBC_INLINE OverflowWriter(OverflowWriteFn<CharT>, void *initial_stack_buffer)
      : initial_stack_buffer(initial_stack_buffer) {}

  LIBC_INLINE int write(WriteBuffer<CharT> &wb,
                        cpp::basic_string_view<CharT> new_str) {
    return overflow_write_resize_buffer(wb, new_str, initial_stack_buffer);
  }
};

template <typename CharT>
struct OverflowWriter<OverflowMode::RUNTIME_DISPATCH, CharT> {
  OverflowWriteFn<CharT> runtime_fn;
  void *state;

  LIBC_INLINE OverflowWriter(OverflowWriteFn<CharT> runtime_fn, void *state)
      : runtime_fn(runtime_fn), state(state) {}

  LIBC_INLINE int write(WriteBuffer<CharT> &wb,
                        cpp::basic_string_view<CharT> new_str) {
    return runtime_fn(wb, new_str, state);
  }
};

// Constructs an `OverflowWriter` that dispatches according to
// `LIBC_COPT_PRINTF_RUNTIME_DISPATCH`.
template <OverflowMode mode, typename CharT>
LIBC_INLINE OverflowWriter<Mode<mode>::value, CharT>
make_overflow_writer(void *state) {
  return OverflowWriter<Mode<mode>::value, CharT>(
      OverflowModeToWriteFn<mode, CharT>::value, state);
}

// Fills the `dest` buffer with `count` copies of `value`.
template <typename CharT>
LIBC_INLINE void fill_buffer(CharT *dest, CharT value, size_t count) {
  if constexpr (sizeof(CharT) == sizeof(unsigned char)) {
    inline_memset(dest, static_cast<unsigned char>(value), count);
  } else {
    for (size_t i = 0; i < count; ++i)
      dest[i] = value;
  }
}

template <OverflowMode mode, typename CharT = char> class Writer final {
  WriteBuffer<CharT> wb;
  size_t chars_written = 0;
  OverflowWriter<mode, CharT> overflow_writer;

  LIBC_INLINE int pad(CharT new_char, size_t length) {
    // First, fill as much of the buffer as possible with the padding char.
    size_t written = 0;
    const size_t buff_space = wb.buff_len - wb.buff_cur;
    // ASSERT: length > buff_space
    if (buff_space > 0) {
      fill_buffer(wb.buff + wb.buff_cur, new_char, buff_space);
      wb.buff_cur += buff_space;
      written = buff_space;
    }

    // Next, overflow write the rest of length using the mini_buff.
    constexpr size_t MINI_BUFF_SIZE = 64;
    CharT mini_buff[MINI_BUFF_SIZE];
    fill_buffer(mini_buff, new_char, MINI_BUFF_SIZE);
    cpp::basic_string_view<CharT> mb_string_view(mini_buff, MINI_BUFF_SIZE);
    while (written + MINI_BUFF_SIZE < length) {
      int result = overflow_writer.write(wb, mb_string_view);
      if (result != WRITE_OK)
        return result;
      written += MINI_BUFF_SIZE;
    }
    cpp::basic_string_view<CharT> mb_substr =
        mb_string_view.substr(0, length - written);
    return overflow_writer.write(wb, mb_substr);
  }

public:
  // Prefer using the factory functions in make_*_writer.h or
  // `make_drop_overflow_writer` below, instead of directly calling this
  // constructor.
  LIBC_INLINE Writer(CharT *buffer, size_t buffer_len,
                     OverflowWriter<mode, CharT> overflow_writer)
      : wb{.buff = buffer, .buff_len = buffer_len},
        overflow_writer(overflow_writer) {}

  // Takes a string, copies it into the buffer if there is space, else passes it
  // to the overflow mechanism to be handled separately.
  LIBC_INLINE int write(cpp::basic_string_view<CharT> new_string) {
    chars_written += new_string.size();
    if (LIBC_LIKELY(wb.buff_cur + new_string.size() <= wb.buff_len)) {
      inline_memcpy(wb.buff + wb.buff_cur, new_string.data(),
                    new_string.size() * sizeof(CharT));
      wb.buff_cur += new_string.size();
      return WRITE_OK;
    }
    return overflow_writer.write(wb, new_string);
  }

  // Takes a char and a length, memsets the next length characters of the buffer
  // if there is space, else calls pad which will loop and call the overflow
  // mechanism on a secondary buffer.
  LIBC_INLINE int write(CharT new_char, size_t length) {
    chars_written += length;

    if (LIBC_LIKELY(wb.buff_cur + length <= wb.buff_len)) {
      fill_buffer(wb.buff + wb.buff_cur, new_char, length);
      wb.buff_cur += length;
      return WRITE_OK;
    }
    return pad(new_char, length);
  }

  // Takes a char, copies it into the buffer if there is space, else passes it
  // to the overflow mechanism to be handled separately.
  LIBC_INLINE int write(CharT new_char) {
    chars_written += 1;
    if (LIBC_LIKELY(wb.buff_cur + 1 <= wb.buff_len)) {
      wb.buff[wb.buff_cur] = new_char;
      wb.buff_cur += 1;
      return WRITE_OK;
    }
    return overflow_writer.write(wb, {&new_char, 1});
  }

  LIBC_INLINE size_t get_chars_written() { return chars_written; }

  LIBC_INLINE WriteBuffer<CharT> &get_write_buffer() { return wb; }
};

// Class-template auto deduction helper.
template <OverflowMode mode, typename CharT>
Writer(CharT *, size_t, OverflowWriter<mode, CharT>) -> Writer<mode, CharT>;

template <typename CharT>
LIBC_INLINE Writer<Mode<OverflowMode::DROP_OVERFLOW>::value, CharT>
make_drop_overflow_writer(CharT *buffer, size_t buffer_len) {
  return Writer(
      buffer, buffer_len,
      make_overflow_writer<OverflowMode::DROP_OVERFLOW, CharT>(nullptr));
}

} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_WRITER_H
