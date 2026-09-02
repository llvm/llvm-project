//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation helpers for Writer<OverflowMode::FLUSH_TO_FILE, CharT>.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_MAKE_FILE_WRITER_H
#define LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_MAKE_FILE_WRITER_H

#include "hdr/types/FILE.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/writer.h"

namespace LIBC_NAMESPACE_DECL {
namespace printf_core {

template <typename CharT>
LIBC_INLINE Writer<Mode<OverflowMode::FLUSH_TO_FILE>::value, CharT>
make_file_writer(CharT *buffer, size_t buffer_len, ::FILE *fp) {
  return Writer(buffer, buffer_len,
                make_overflow_writer<OverflowMode::FLUSH_TO_FILE, CharT>(fp));
}

} // namespace printf_core

namespace internal {
#ifndef LIBC_COPT_STDIO_USE_SYSTEM_FILE
LIBC_INLINE int ferror_unlocked(FILE *f) {
  return reinterpret_cast<LIBC_NAMESPACE::File *>(f)->error_unlocked();
}

LIBC_INLINE void flockfile(FILE *f) {
  reinterpret_cast<LIBC_NAMESPACE::File *>(f)->lock();
}

LIBC_INLINE void funlockfile(FILE *f) {
  reinterpret_cast<LIBC_NAMESPACE::File *>(f)->unlock();
}

LIBC_INLINE FileIOResult fwrite_unlocked(const void *ptr, size_t size,
                                         size_t nmemb, FILE *f) {
  return reinterpret_cast<LIBC_NAMESPACE::File *>(f)->write_unlocked(
      ptr, size * nmemb);
}
#else  // defined(LIBC_COPT_STDIO_USE_SYSTEM_FILE)
LIBC_INLINE int ferror_unlocked(::FILE *f) { return ::ferror_unlocked(f); }

LIBC_INLINE void flockfile(::FILE *f) { ::flockfile(f); }

LIBC_INLINE void funlockfile(::FILE *f) { ::funlockfile(f); }

LIBC_INLINE FileIOResult fwrite_unlocked(const void *ptr, size_t size,
                                         size_t nmemb, ::FILE *f) {
  // Need to use system errno in this case, as system write will set this errno
  // which we need to propagate back into our code. fwrite only modifies errno
  // if there was an error, and errno may have previously been nonzero. Only
  // return errno if there was an error.
  size_t members_written = ::fwrite_unlocked(ptr, size, nmemb, f);
  return {members_written, members_written == nmemb ? 0 : errno};
}
#endif // LIBC_COPT_STDIO_USE_SYSTEM_FILE
} // namespace internal

namespace printf_core {

LIBC_INLINE int write_to_file_unlocked(cpp::string_view str,
                                       ::FILE *target_file) {
  if (str.size() == 0)
    return WRITE_OK;

  auto write_result = internal::fwrite_unlocked(str.data(), sizeof(char),
                                                str.size(), target_file);
  // Propagate actual system error in FileIOResult.
  if (write_result.has_error())
    return -write_result.error;

  // In case short write occured or error was not set on FileIOResult for some
  // reason.
  if (write_result.value != str.size() ||
      internal::ferror_unlocked(target_file))
    return FILE_WRITE_ERROR;

  return WRITE_OK;
}

// Handles overflow by flushing the current contents of `wb` and `new_str` to
// the `FILE` pointed to by `fp`.
template <typename CharT>
LIBC_INLINE int
overflow_write_flush_to_file(WriteBuffer<CharT> &wb,
                             cpp::basic_string_view<CharT> new_str, void *fp) {
  ::FILE *target_file = reinterpret_cast<::FILE *>(fp);
  int retval;

  retval = write_to_file_unlocked({wb.buff, wb.buff_cur}, target_file);
  if (retval < 0)
    return retval;
  wb.buff_cur = 0;

  retval = write_to_file_unlocked(new_str, target_file);
  if (retval < 0)
    return retval;

  return WRITE_OK;
}

} // namespace printf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_PRINTF_CORE_MAKE_FILE_WRITER_H
