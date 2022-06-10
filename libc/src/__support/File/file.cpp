//===--- Implementation of a platform independent file data structure -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "file.h"

#include "src/__support/CPP/ArrayRef.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

namespace __llvm_libc {

size_t File::write_unlocked(const void *data, size_t len) {
  if (!write_allowed()) {
    errno = EBADF;
    err = true;
    return 0;
  }

  prev_op = FileOp::WRITE;

  if (bufmode == _IOFBF) { // fully buffered
    return write_unlocked_fbf(data, len);
  } else if (bufmode == _IOLBF) { // line buffered
    return write_unlocked_lbf(data, len);
  } else /*if (bufmode == _IONBF) */ { // unbuffered
    size_t ret_val = write_unlocked_nbf(data, len);
    flush_unlocked();
    return ret_val;
  }
}

size_t File::write_unlocked_nbf(const void *data, size_t len) {
  if (pos > 0) { // If the buffer is not empty
    // Flush the buffer
    const size_t write_size = pos;
    size_t bytes_written = platform_write(this, buf, write_size);
    pos = 0; // Buffer is now empty so reset pos to the beginning.
    // If less bytes were written than expected, then an error occurred.
    if (bytes_written < write_size) {
      err = true;
      return 0; // No bytes from data were written, so return 0.
    }
  }

  size_t written = platform_write(this, data, len);
  if (written < len)
    err = true;
  return written;
}

size_t File::write_unlocked_fbf(const void *data, size_t len) {
  const size_t init_pos = pos;
  const size_t bufspace = bufsize - pos;

  // If data is too large to be buffered at all, then just write it unbuffered.
  if (len > bufspace + bufsize)
    return write_unlocked_nbf(data, len);

  // we split |data| (conceptually) using the split point. Then we handle the
  // two pieces separately.
  const size_t split_point = len < bufspace ? len : bufspace;

  // The primary piece is the piece of |data| we want to write to the buffer
  // before flushing. It will always fit into the buffer, since the split point
  // is defined as being min(len, bufspace), and it will always exist if len is
  // non-zero.
  cpp::ArrayRef<uint8_t> primary(data, split_point);

  // The second piece is the remainder of |data|. It is written to the buffer if
  // it fits, or written directly to the output if it doesn't. If the primary
  // piece fits entirely in the buffer, the remainder may be nothing.
  cpp::ArrayRef<uint8_t> remainder(
      static_cast<const uint8_t *>(data) + split_point, len - split_point);

  cpp::MutableArrayRef<uint8_t> bufref(buf, bufsize);

  // Copy the first piece into the buffer.
  // TODO: Replace the for loop below with a call to internal memcpy.
  for (size_t i = 0; i < primary.size(); ++i)
    bufref[pos + i] = primary[i];
  pos += primary.size();

  // If there is no remainder, we can return early, since the first piece has
  // fit completely into the buffer.
  if (remainder.size() == 0)
    return len;

  // We need to flush the buffer now, since there is still data and the buffer
  // is full.
  const size_t write_size = pos;
  size_t bytes_written = platform_write(this, buf, write_size);
  pos = 0; // Buffer is now empty so reset pos to the beginning.
  // If less bytes were written than expected, then an error occurred. Return
  // the number of bytes that have been written from |data|.
  if (bytes_written < write_size) {
    err = true;
    return bytes_written <= init_pos ? 0 : bytes_written - init_pos;
  }

  // The second piece is handled basically the same as the first, although we
  // know that if the second piece has data in it then the buffer has been
  // flushed, meaning that pos is always 0.
  if (remainder.size() < bufsize) {
    // TODO: Replace the for loop below with a call to internal memcpy.
    for (size_t i = 0; i < remainder.size(); ++i)
      bufref[i] = remainder[i];
    pos = remainder.size();
  } else {
    size_t bytes_written =
        platform_write(this, remainder.data(), remainder.size());

    // If less bytes were written than expected, then an error occurred. Return
    // the number of bytes that have been written from |data|.
    if (bytes_written < remainder.size()) {
      err = true;
      return primary.size() + bytes_written;
    }
  }

  return len;
}

size_t File::write_unlocked_lbf(const void *data, size_t len) {
  const size_t init_pos = pos;
  const size_t bufspace = bufsize - pos;

  constexpr char NEWLINE_CHAR = '\n';
  size_t last_newline = len;
  for (size_t i = len - 1; i > 0; --i) {
    if (static_cast<const char *>(data)[i] == NEWLINE_CHAR) {
      last_newline = i;
      break;
    }
  }

  // If there is no newline, treat this as fully buffered.
  if (last_newline == len) {
    return write_unlocked_fbf(data, len);
  }

  // we split |data| (conceptually) using the split point. Then we handle the
  // two pieces separately.
  const size_t split_point = last_newline + 1;

  // The primary piece is everything in |data| up to the newline. It's written
  // unbuffered to the output.
  cpp::ArrayRef<uint8_t> primary(data, split_point);

  // The second piece is the remainder of |data|. It is written fully buffered,
  // meaning it may stay in the buffer if it fits.
  cpp::ArrayRef<uint8_t> remainder(
      static_cast<const uint8_t *>(data) + split_point, len - split_point);

  size_t written = 0;

  written = write_unlocked_nbf(primary.data(), primary.size());
  if (written < primary.size()) {
    err = true;
    return written;
  }

  flush_unlocked();

  written += write_unlocked_fbf(remainder.data(), remainder.size());
  if (written < primary.size() + remainder.size()) {
    err = true;
    return written;
  }

  return len;
}

size_t File::read_unlocked(void *data, size_t len) {
  if (!read_allowed()) {
    errno = EBADF;
    err = true;
    return 0;
  }

  prev_op = FileOp::READ;

  cpp::MutableArrayRef<uint8_t> bufref(buf, bufsize);
  cpp::MutableArrayRef<uint8_t> dataref(data, len);

  // Because read_limit is always greater than equal to pos,
  // available_data is never a wrapped around value.
  size_t available_data = read_limit - pos;
  if (len <= available_data) {
    // TODO: Replace the for loop below with a call to internal memcpy.
    for (size_t i = 0; i < len; ++i)
      dataref[i] = bufref[i + pos];
    pos += len;
    return len;
  }

  // Copy all of the available data.
  // TODO: Replace the for loop with a call to internal memcpy.
  for (size_t i = 0; i < available_data; ++i)
    dataref[i] = bufref[i + pos];
  read_limit = pos = 0; // Reset the pointers.

  size_t to_fetch = len - available_data;
  if (to_fetch > bufsize) {
    size_t fetched_size = platform_read(this, data, to_fetch);
    if (fetched_size < to_fetch) {
      if (errno == 0)
        eof = true;
      else
        err = true;
      return available_data + fetched_size;
    }
    return len;
  }

  // Fetch and buffer another buffer worth of data.
  size_t fetched_size = platform_read(this, buf, bufsize);
  read_limit += fetched_size;
  size_t transfer_size = fetched_size >= to_fetch ? to_fetch : fetched_size;
  for (size_t i = 0; i < transfer_size; ++i)
    dataref[i] = bufref[i];
  pos += transfer_size;
  if (fetched_size < to_fetch) {
    if (errno == 0)
      eof = true;
    else
      err = true;
  }
  return transfer_size + available_data;
}

int File::seek(long offset, int whence) {
  FileLock lock(this);
  if (prev_op == FileOp::WRITE && pos > 0) {
    size_t transferred_size = platform_write(this, buf, pos);
    if (transferred_size < pos) {
      err = true;
      return -1;
    }
  } else if (prev_op == FileOp::READ && whence == SEEK_CUR) {
    // More data could have been read out from the platform file than was
    // required. So, we have to adjust the offset we pass to platform seek
    // function. Note that read_limit >= pos is always true.
    offset -= (read_limit - pos);
  }
  pos = read_limit = 0;
  prev_op = FileOp::SEEK;
  // Reset the eof flag as a seek might move the file positon to some place
  // readable.
  eof = false;
  return platform_seek(this, offset, whence);
}

int File::flush_unlocked() {
  if (prev_op == FileOp::WRITE && pos > 0) {
    size_t transferred_size = platform_write(this, buf, pos);
    if (transferred_size < pos) {
      err = true;
      return -1;
    }
    pos = 0;
    return platform_flush(this);
  }
  // TODO: Add POSIX behavior for input streams.
  return 0;
}

int File::close() {
  {
    FileLock lock(this);
    if (prev_op == FileOp::WRITE && pos > 0) {
      size_t transferred_size = platform_write(this, buf, pos);
      if (transferred_size < pos) {
        err = true;
        return -1;
      }
    }
    if (platform_close(this) != 0)
      return -1;
    if (own_buf)
      free(buf);
  }
  free(this);
  return 0;
}

void File::set_buffer(void *buffer, size_t size, bool owned) {
  if (own_buf)
    free(buf);
  buf = buffer;
  bufsize = size;
  own_buf = owned;
}

File::ModeFlags File::mode_flags(const char *mode) {
  // First character in |mode| should be 'a', 'r' or 'w'.
  if (*mode != 'a' && *mode != 'r' && *mode != 'w')
    return 0;

  // There should be exaclty one main mode ('a', 'r' or 'w') character.
  // If there are more than one main mode characters listed, then
  // we will consider |mode| as incorrect and return 0;
  int main_mode_count = 0;

  ModeFlags flags = 0;
  for (; *mode != '\0'; ++mode) {
    switch (*mode) {
    case 'r':
      flags |= static_cast<ModeFlags>(OpenMode::READ);
      ++main_mode_count;
      break;
    case 'w':
      flags |= static_cast<ModeFlags>(OpenMode::WRITE);
      ++main_mode_count;
      break;
    case '+':
      flags |= static_cast<ModeFlags>(OpenMode::PLUS);
      break;
    case 'b':
      flags |= static_cast<ModeFlags>(ContentType::BINARY);
      break;
    case 'a':
      flags |= static_cast<ModeFlags>(OpenMode::APPEND);
      ++main_mode_count;
      break;
    case 'x':
      flags |= static_cast<ModeFlags>(CreateType::EXCLUSIVE);
      break;
    default:
      return 0;
    }
  }

  if (main_mode_count != 1)
    return 0;

  return flags;
}

} // namespace __llvm_libc
