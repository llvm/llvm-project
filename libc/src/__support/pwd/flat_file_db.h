//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Generic flat-file database template engine.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_PWD_FLAT_FILE_DB_H
#define LLVM_LIBC_SRC___SUPPORT_PWD_FLAT_FILE_DB_H

#include "hdr/errno_macros.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/off_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/functional.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/span.h"
#include "src/__support/File/file.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/pwd/dynamic_buffer.h"

namespace LIBC_NAMESPACE_DECL {
namespace pwd {

// Struct to hold the result of a line read operation.
struct ReadLineResult {
  size_t bytes_read;
  size_t raw_bytes_consumed;
  bool truncated;
  // True only when zero bytes were read because the stream was already at EOF.
  // A final line without a trailing newline returns bytes_read > 0 and
  // eof == false; the following call returns bytes_read == 0 and eof == true.
  bool eof;
};

// Parses a record in place and fills entry.
// If the buffer is too small for auxiliary structures (such as pointer arrays),
// specializations must return Error(ERANGE) prior to modifying the buffer.
template <typename EntryType>
ErrorOr<void> parse_line(cpp::span<char> buffer, size_t line_len,
                         EntryType *entry);

// Generic flat colon-delimited database engine.
template <typename EntryType> class FlatFileDatabase {
public:
  using Matcher = cpp::function<bool(const EntryType &)>;

private:
  const char *file_path;
  File *file = nullptr;
  off_t current_offset = 0;
  off_t last_line_start = 0;

  // Reads a single line from the given file into the provided buffer, stripping
  // any trailing '\n' and ensuring the result is null-terminated. A line too
  // long for the buffer is reported as truncated.
  //
  // Note: POSIX getline/getdelim cannot be used here because user database
  // iteration and lookups must operate in-place within a fixed, bounded buffer
  // without dynamic heap allocations during getnext. See read_line_growing for
  // the variant used by the non-reentrant interfaces, which own their buffer
  // and may grow it.
  LIBC_INLINE static ErrorOr<ReadLineResult> read_line(File *f,
                                                       cpp::span<char> buf) {
    if (!f)
      return Error(EINVAL);
    if (buf.size() < 2)
      return Error(ERANGE);

    File::FileLock lock(f);
    size_t bytes_read = 0;
    size_t raw_bytes_consumed = 0;
    FileIOResult result(0);
    bool truncated = false;

    for (char &ch : buf.first(buf.size() - 1)) {
      result = f->read_unlocked(&ch, 1);
      if (result.has_error())
        return Error(result.error);
      if (result.value != 1)
        break;
      ++bytes_read;
      ++raw_bytes_consumed;
      if (ch == '\n')
        break;
    }

    bool eof = (bytes_read == 0);

    auto read_span = buf.first(bytes_read);
    if (result.value == 1 && !read_span.empty() && read_span.back() != '\n') {
      char c = '\0';
      while (true) {
        result = f->read_unlocked(&c, 1);
        if (result.has_error())
          return Error(result.error);
        if (result.value != 1)
          break;
        ++raw_bytes_consumed;
        if (c == '\n')
          break;
        truncated = true;
      }
    }

    if (f->error_unlocked())
      return Error(EIO);

    // If the line ended with a newline, strip it.
    if (!read_span.empty() && read_span.back() == '\n')
      --bytes_read;

    buf[bytes_read] = '\0';
    return ReadLineResult{bytes_read, raw_bytes_consumed, truncated, eof};
  }

  // Reads a single line into a caller-owned buffer, growing it as needed so
  // that arbitrarily long records can be read. Otherwise behaves as read_line;
  // the result is never truncated.
  LIBC_INLINE static ErrorOr<ReadLineResult>
  read_line_growing(File *f, DynamicBuffer &buf) {
    if (!f)
      return Error(EINVAL);

    File::FileLock lock(f);
    size_t bytes_read = 0;

    while (true) {
      // One byte for the character about to be read, one for the terminator.
      if (bytes_read > cpp::numeric_limits<size_t>::max() - 2 ||
          (bytes_read + 2 > buf.capacity() && !buf.reserve(bytes_read + 2))) {
        char c = '\0';
        while (true) {
          FileIOResult drain = f->read_unlocked(&c, 1);
          if (drain.has_error() || drain.value != 1 || c == '\n')
            break;
        }
        return Error(ENOMEM);
      }

      char ch = '\0';
      FileIOResult result = f->read_unlocked(&ch, 1);
      if (result.has_error())
        return Error(result.error);
      if (result.value != 1)
        break;

      buf.span()[bytes_read++] = ch;
      if (ch == '\n')
        break;
    }

    if (f->error_unlocked())
      return Error(EIO);

    bool eof = (bytes_read == 0);
    size_t raw_bytes_consumed = bytes_read;

    // If the line ended with a newline, strip it.
    if (bytes_read > 0 && buf.span()[bytes_read - 1] == '\n')
      --bytes_read;

    buf.span()[bytes_read] = '\0';
    return ReadLineResult{bytes_read, raw_bytes_consumed, /*truncated=*/false,
                          eof};
  }

public:
  LIBC_INLINE constexpr explicit FlatFileDatabase(const char *path)
      : file_path(path) {}

  FlatFileDatabase(const FlatFileDatabase &) = delete;
  FlatFileDatabase &operator=(const FlatFileDatabase &) = delete;

  // Sets or overrides the file path for database operations.
  LIBC_INLINE void set_path(const char *path) {
    if (!path)
      return;
    if (file) {
      file->close();
      file = nullptr;
    }
    file_path = path;
    current_offset = 0;
    last_line_start = 0;
  }

  // Opens or rewinds the database file stream.
  LIBC_INLINE ErrorOr<void> setdb() {
    current_offset = 0;
    last_line_start = 0;
    if (!file) {
      auto result = openfile(file_path, "r");
      if (!result.has_value())
        return Error(result.error());
      file = result.value();
      return {};
    }
    auto result = file->seek(0, SEEK_SET);
    if (!result.has_value())
      return Error(result.error());
    file->clearerr();
    return {};
  }

  // Closes the database file stream.
  LIBC_INLINE ErrorOr<void> enddb() {
    current_offset = 0;
    last_line_start = 0;
    if (file) {
      int result = file->close();
      file = nullptr;
      if (result != 0)
        return Error(result);
    }
    return {};
  }

  // Reads and parses the next record from the database into a fixed buffer.
  // Returns true if an entry was read, false if EOF was reached, or an Error on
  // failure. Blank lines are skipped. A record that does not fit in the buffer
  // is reported as ERANGE.
  LIBC_INLINE ErrorOr<bool> getnext(EntryType *entry, cpp::span<char> buffer) {
    if (!entry)
      return Error(EINVAL);

    if (!file) {
      auto res = setdb();
      if (!res.has_value())
        return Error(res.error());
    }

    while (true) {
      last_line_start = current_offset;
      auto result = read_line(file, buffer);
      if (!result.has_value())
        return Error(result.error());

      ReadLineResult res = result.value();
      current_offset += static_cast<off_t>(res.raw_bytes_consumed);
      if (res.eof)
        return false; // EOF

      // Skip blank lines.
      if (res.bytes_read == 0)
        continue;

      if (res.truncated)
        return Error(ERANGE);

      auto parse_res = parse_line<EntryType>(buffer, res.bytes_read, entry);
      if (!parse_res.has_value())
        return Error(parse_res.error());
      return true;
    }
  }

  // Reads and parses the next record from the database into a caller-owned
  // buffer, growing it as needed. Behaves as the fixed-buffer overload except
  // that a long record grows the buffer rather than producing ERANGE.
  LIBC_INLINE ErrorOr<bool> getnext(EntryType *entry, DynamicBuffer &buffer) {
    if (!entry)
      return Error(EINVAL);

    if (!file) {
      auto res = setdb();
      if (!res.has_value())
        return Error(res.error());
    }

    while (true) {
      last_line_start = current_offset;
      auto result = read_line_growing(file, buffer);
      if (!result.has_value())
        return Error(result.error());

      ReadLineResult res = result.value();
      current_offset += static_cast<off_t>(res.raw_bytes_consumed);
      if (res.eof)
        return false; // EOF

      // Skip blank lines.
      if (res.bytes_read == 0)
        continue;

      while (true) {
        auto parse_res =
            parse_line<EntryType>(buffer.span(), res.bytes_read, entry);
        if (parse_res.has_value())
          return true;
        if (parse_res.error() != ERANGE)
          return Error(parse_res.error());
        if (!buffer.grow())
          return Error(ENOMEM);
      }
    }
  }

  // Searches for a record matching a given predicate. Returns true if the
  // entry was found, false if it's missing, or an Error if lookup failed.
  //
  // Per POSIX, ERANGE is reported only if the matched entry does not fit in the
  // caller's buffer; unrelated preceding records larger than buffer are
  // skipped.
  LIBC_INLINE ErrorOr<bool> lookup(const Matcher &matcher, EntryType *entry,
                                   cpp::span<char> buffer) {
    if (!entry)
      return Error(EINVAL);

    auto res = setdb();
    if (!res.has_value())
      return Error(res.error());

    ScopedDynamicBuffer scratch_buf;
    while (true) {
      auto next_res = getnext(entry, buffer);
      if (next_res.has_value()) {
        if (!next_res.value())
          return false; // EOF without match
        if (matcher(*entry))
          return true;
        continue;
      }

      if (next_res.error() != ERANGE)
        return Error(next_res.error());

      // The record at last_line_start exceeded buffer. Check whether it is
      // actually the target entry before reporting ERANGE. Use a stack-local
      // scratch_entry so we do not leave dangling pointers in the caller's
      // *entry when scratch_buf goes out of scope.
      auto seek_res = file->seek(last_line_start, SEEK_SET);
      if (!seek_res.has_value())
        return Error(seek_res.error());
      file->clearerr();
      current_offset = last_line_start;

      EntryType scratch_entry{};
      auto dyn_res = getnext(&scratch_entry, scratch_buf);
      if (!dyn_res.has_value())
        return Error(dyn_res.error());
      if (!dyn_res.value())
        return false;
      if (matcher(scratch_entry))
        return Error(ERANGE);
    }
  }

  // As above, but reads into a caller-owned buffer that grows to fit long
  // records instead of reporting ERANGE.
  LIBC_INLINE ErrorOr<bool> lookup(const Matcher &matcher, EntryType *entry,
                                   DynamicBuffer &buffer) {
    if (!entry)
      return Error(EINVAL);

    auto res = setdb();
    if (!res.has_value())
      return Error(res.error());

    while (true) {
      auto next_res = getnext(entry, buffer);
      if (!next_res.has_value())
        return Error(next_res.error());
      if (!next_res.value())
        return false; // EOF without match
      if (matcher(*entry))
        return true;
    }
  }
};

// RAII wrapper around FlatFileDatabase for stack-local database operations.
// Automatically closes the database file stream on destruction.
template <typename EntryType>
class ScopedFlatFileDatabase : public FlatFileDatabase<EntryType> {
public:
  using FlatFileDatabase<EntryType>::FlatFileDatabase;

  LIBC_INLINE ~ScopedFlatFileDatabase() { this->enddb(); }
};

} // namespace pwd
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_PWD_FLAT_FILE_DB_H
