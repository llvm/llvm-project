//===--- A platform independent Dir class ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FILE_DIR_H
#define LLVM_LIBC_SRC_SUPPORT_FILE_DIR_H

#include "src/__support/CPP/ArrayRef.h"
#include "src/__support/threads/mutex.h"

#include <dirent.h>
#include <stdlib.h>

namespace __llvm_libc {

// Platform specific function which will open the directory |name|
// and return its file descriptor. Upon failure, this function sets the errno
// value as suitable.
int platform_opendir(const char *name);

// Platform specific function which will close the directory with
// file descriptor |fd|. Returns true on success, false on failure.
bool platform_closedir(int fd);

// Platform specific function which will fetch dirents in to buffer.
// Returns the number of bytes written into buffer
size_t platform_fetch_dirents(int fd, cpp::MutableArrayRef<uint8_t> buffer);

// This class is designed to allow implementation of the POSIX dirent.h API.
// By itself, it is platform independent but calls platform specific
// functions to perform OS operations.
class Dir {
  static constexpr size_t BUFSIZE = 1024;
  int fd;
  size_t readptr = 0;  // The current read pointer.
  size_t fillsize = 0; // The number of valid bytes availabe in the buffer.

  // This is a buffer of struct dirent values which will be fetched
  // from the OS. Since the d_name of struct dirent can be of a variable
  // size, we store the data in a byte array.
  uint8_t buffer[BUFSIZE];

  Mutex mutex;

public:
  // A directory is to be opened by the static method open and closed
  // by the close method. So, all constructors and destructor are declared
  // as deleted.
  Dir() = delete;
  Dir(const Dir &) = delete;
  ~Dir() = delete;

  Dir &operator=(const Dir &) = delete;

  static Dir *open(const char *path);

  struct ::dirent *read();

  int close();

  int getfd() { return fd; }
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_FILE_DIR_H
