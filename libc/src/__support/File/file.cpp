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

  cpp::ArrayRef<uint8_t> dataref(data, len);
  cpp::MutableArrayRef<uint8_t> bufref(buf, bufsize);

  const size_t used = pos;
  const size_t bufspace = bufsize - pos;
  const size_t write_size = bufspace > len ? len : bufspace;
  // TODO: Replace the for loop below with a call to internal memcpy.
  for (size_t i = 0; i < write_size; ++i)
    bufref[pos + i] = dataref[i];
  pos += write_size;
  if (len < bufspace)
    return len;

  // If the control reaches beyond this point, it means that |data|
  // is more than what can be accomodated in the buffer. So, we first
  // flush out the buffer.
  size_t bytes_written = platform_write(this, buf, bufsize);
  pos = 0; // Buffer is now empty so reset pos to the beginning.
  if (bytes_written < bufsize) {
    err = true;
    // If less bytes were written than expected, then there are two
    // possibilities.
    // 1. None of the bytes from |data| were flushed out.
    if (bytes_written <= used)
      return 0;
    // 2. Some of the bytes from |data| were written
    return bytes_written - used;
  }

  // If the remaining bytes from |data| can fit in the buffer, write
  // into it. Else, write it directly to the platform stream.
  size_t remaining = len - write_size;
  if (remaining <= len) {
    // TODO: Replace the for loop below with a call to internal memcpy.
    for (size_t i = 0; i < remaining; ++i)
      bufref[i] = dataref[i];
    pos += remaining;
    return len;
  }

  size_t transferred =
      platform_write(this, dataref.data() + write_size, remaining);
  if (transferred < remaining) {
    err = true;
    return write_size + transferred;
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

int File::flush() {
  FileLock lock(this);
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
