//===---------- Linux implementation of the getauxval function ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/auxv/getauxval.h"
#include "config/linux/app.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h"
#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/prctl/prctl.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"

#include <linux/auxvec.h> // AT_NULL
#include <sys/mman.h>     // MAP_FAILED, MAP_PRIVATE, MAP_ANONYMOUS

namespace LIBC_NAMESPACE {

constexpr size_t AUXV_MMAP_SIZE = sizeof(AuxEntryType) * 64;

struct ErrorNo {
  ErrorNo() : errno_backup(libc_errno), failure(false) {}
  ~ErrorNo() { libc_errno = failure ? ENOENT : errno_backup; }
  void mark_failure() { failure = true; }
  int errno_backup;
  bool failure;
};

struct TempAuxv {
  TempAuxv() : ptr(nullptr) {}
  ~TempAuxv() {
    if (ptr != nullptr)
      munmap(ptr, AUXV_MMAP_SIZE);
  }
  AuxEntryType *ptr;
};

struct TempFD {
  TempFD(const char *path) : fd(open(path, O_RDONLY | O_CLOEXEC)) {}
  ~TempFD() {
    if (fd >= 0)
      close(fd);
  }
  operator int() const { return fd; }
  int fd;
};

static AuxEntryType read_entry(int fd) {
  AuxEntryType buf;
  ssize_t size = sizeof(AuxEntryType);
  while (size > 0) {
    ssize_t ret = read(fd, &buf, size);
    if (ret < 0) {
      if (libc_errno == EINTR)
        continue;
      buf.id = AT_NULL;
      buf.value = AT_NULL;
      break;
    }
    size -= ret;
  }
  return buf;
}

LLVM_LIBC_FUNCTION(unsigned long, getauxval, (unsigned long id)) {
  // if llvm-libc's loader is applied, app.auxv_ptr should have been
  // initialized, then we can directly get the auxillary vector
  const AuxEntryType *auxv_ptr = app.auxv_ptr;
  ErrorNo errno_holder;
  TempAuxv temp_auxv;

  // Compatible layer for the overlay mode.
  // first check if PR_GET_AUXV is supported. Unfortunately, this is not
  // supported until Linux 6.0+. We check this syscall first since it may be
  // the case that /proc is not mounted.
#ifdef PR_GET_AUXV
  do {
    if (LIBC_UNLIKELY(auxv_ptr == nullptr)) {
      // no need to backup errno: once it has error, we will modify it
      void *ptr = mmap(nullptr, AUXV_MMAP_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

      if (LIBC_UNLIKELY(ptr == MAP_FAILED))
        break;

      temp_auxv.ptr = reinterpret_cast<AuxEntryType *>(ptr);
      for (size_t i = 0; i < AUXV_MMAP_SIZE / sizeof(AuxEntryType); ++i) {
        temp_auxv.ptr[i].id = AT_NULL;
        temp_auxv.ptr[i].value = AT_NULL;
      }

      // We keeps the last entry to be AT_NULL, so we can always terminate the
      // loop.
      int ret =
          prctl(PR_GET_AUXV, reinterpret_cast<unsigned long>(temp_auxv.ptr),
                AUXV_MMAP_SIZE - sizeof(AuxEntryType), 0, 0);
      if (ret < 0)
        break;

      auxv_ptr = temp_auxv.ptr;
    }
  } while (0);
#endif

  if (LIBC_LIKELY(auxv_ptr != nullptr)) {
    for (; auxv_ptr->id != AT_NULL; ++auxv_ptr) {
      if (auxv_ptr->id == id)
        return auxv_ptr->value;
    }
    errno_holder.mark_failure();
    return 0;
  }

  // now attempt to read from /proc/self/auxv
  do {
    TempFD fd{"/proc/self/auxv"};
    if (fd < 0)
      break;

    while (true) {
      AuxEntryType buf = read_entry(fd);
      if (buf.id == AT_NULL)
        break;
      if (buf.id == id)
        return buf.value;
    }

  } while (0);

  errno_holder.mark_failure();
  return 0;
}

} // namespace LIBC_NAMESPACE
