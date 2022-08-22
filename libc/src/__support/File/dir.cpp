//===--- Implementation of a platform independent Dir data structure ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dir.h"

#include <stdlib.h>

namespace __llvm_libc {

Dir *Dir::open(const char *path) {
  int fd = platform_opendir(path);
  if (fd < 0)
    return nullptr;

  Dir *dir = reinterpret_cast<Dir *>(malloc(sizeof(Dir)));
  dir->fd = fd;
  dir->readptr = 0;
  dir->fillsize = 0;
  Mutex::init(&dir->mutex, false, false, false);

  return dir;
}

struct ::dirent *Dir::read() {
  MutexLock lock(&mutex);
  if (readptr >= fillsize) {
    fillsize = platform_fetch_dirents(fd, buffer);
    if (fillsize == 0)
      return nullptr;
    readptr = 0;
  }
  struct ::dirent *d = reinterpret_cast<struct ::dirent *>(buffer + readptr);
#ifdef __unix__
  // The d_reclen field is available on Linux but not required by POSIX.
  readptr += d->d_reclen;
#else
  // Other platforms have to implement how the read pointer is to be updated.
#error "DIR read pointer update is missing."
#endif
  return d;
}

int Dir::close() {
  {
    MutexLock lock(&mutex);
    if (!platform_closedir(fd))
      return -1;
  }
  free(this);
  return 0;
}

} // namespace __llvm_libc
