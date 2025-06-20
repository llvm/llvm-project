//===-- Implementation file for getauxval function --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/auxv/getauxval.h"
#include "config/app.h"
#include "hdr/fcntl_macros.h"
#include "src/__support/OSUtil/fcntl.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <linux/auxvec.h>

// for guarded initialization
#include "src/__support/threads/callonce.h"
#include "src/__support/threads/linux/futex_word.h"

// -----------------------------------------------------------------------------
// TODO: This file should not include other public libc functions. Calling other
// public libc functions is an antipattern within LLVM-libc. This needs to be
// cleaned up. DO NOT COPY THIS.
// -----------------------------------------------------------------------------

// for mallocing the global auxv
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"

// for reading /proc/self/auxv
#include "src/sys/prctl/prctl.h"
#include "src/unistd/read.h"

// getauxval will work either with or without __cxa_atexit support.
// In order to detect if __cxa_atexit is supported, we define a weak symbol.
// We prefer __cxa_atexit as it is always defined as a C symbol whileas atexit
// may not be created via objcopy yet. Also, for glibc, atexit is provided via
// libc_nonshared.a rather than libc.so. So, it is may not be made ready for
// overlay builds.
extern "C" [[gnu::weak]] int __cxa_atexit(void (*callback)(void *),
                                          void *payload, void *);

namespace LIBC_NAMESPACE_DECL {

constexpr static size_t MAX_AUXV_ENTRIES = 64;

// Helper to recover or set errno
class AuxvErrnoGuard {
public:
  AuxvErrnoGuard() : saved(libc_errno), failure(false) {}
  ~AuxvErrnoGuard() { libc_errno = failure ? ENOENT : saved; }
  void mark_failure() { failure = true; }

private:
  int saved;
  bool failure;
};

// Helper to manage the memory
static AuxEntry *auxv = nullptr;

class AuxvMMapGuard {
public:
  constexpr static size_t AUXV_MMAP_SIZE = sizeof(AuxEntry) * MAX_AUXV_ENTRIES;

  AuxvMMapGuard()
      : ptr(LIBC_NAMESPACE::mmap(nullptr, AUXV_MMAP_SIZE,
                                 PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) {}
  ~AuxvMMapGuard() {
    if (ptr != MAP_FAILED)
      LIBC_NAMESPACE::munmap(ptr, AUXV_MMAP_SIZE);
  }
  void submit_to_global() {
    // atexit may fail, we do not set it to global in that case.
    int ret = __cxa_atexit(
        [](void *) {
          LIBC_NAMESPACE::munmap(auxv, AUXV_MMAP_SIZE);
          auxv = nullptr;
        },
        nullptr, nullptr);

    if (ret != 0)
      return;

    auxv = reinterpret_cast<AuxEntry *>(ptr);
    ptr = MAP_FAILED;
  }
  bool allocated() const { return ptr != MAP_FAILED; }
  void *get() const { return ptr; }

private:
  void *ptr;
};

class AuxvFdGuard {
public:
  AuxvFdGuard() {
    auto result = internal::open("/proc/self/auxv", O_RDONLY | O_CLOEXEC);
    if (!result.has_value())
      fd = -1;

    fd = result.value();
  }
  ~AuxvFdGuard() {
    if (fd != -1)
      internal::close(fd);
  }
  bool valid() const { return fd != -1; }
  int get() const { return fd; }

private:
  int fd;
};

static void initialize_auxv_once(void) {
  // If we cannot get atexit, we cannot register the cleanup function.
  if (&__cxa_atexit == nullptr)
    return;

  AuxvMMapGuard mmap_guard;
  if (!mmap_guard.allocated())
    return;
  auto *ptr = reinterpret_cast<AuxEntry *>(mmap_guard.get());

  // We get one less than the max size to make sure the search always
  // terminates. MMAP private pages are zeroed out already.
  size_t available_size = AuxvMMapGuard::AUXV_MMAP_SIZE - sizeof(AuxEntryType);
  // PR_GET_AUXV is only available on Linux kernel 6.1 and above. If this is not
  // defined, we direcly fall back to reading /proc/self/auxv. In case the libc
  // is compiled and run on separate kernels, we also check the return value of
  // prctl.
#ifdef PR_GET_AUXV
  int ret = prctl(PR_GET_AUXV, reinterpret_cast<unsigned long>(ptr),
                  available_size, 0, 0);
  if (ret >= 0) {
    mmap_guard.submit_to_global();
    return;
  }
#endif
  AuxvFdGuard fd_guard;
  if (!fd_guard.valid())
    return;
  auto *buf = reinterpret_cast<char *>(ptr);
  libc_errno = 0;
  bool error_detected = false;
  // Read until we use up all the available space or we finish reading the file.
  while (available_size != 0) {
    ssize_t bytes_read =
        LIBC_NAMESPACE::read(fd_guard.get(), buf, available_size);
    if (bytes_read <= 0) {
      if (libc_errno == EINTR)
        continue;
      // Now, we either have an non-recoverable error or we have reached the end
      // of the file. Mark `error_detected` accordingly.
      if (bytes_read == -1)
        error_detected = true;
      break;
    }
    buf += bytes_read;
    available_size -= bytes_read;
  }
  // If we get out of the loop without an error, the auxv is ready.
  if (!error_detected)
    mmap_guard.submit_to_global();
}

static AuxEntry read_entry(int fd) {
  AuxEntry buf;
  size_t size = sizeof(AuxEntry);
  char *ptr = reinterpret_cast<char *>(&buf);
  while (size > 0) {
    ssize_t ret = LIBC_NAMESPACE::read(fd, ptr, size);
    if (ret < 0) {
      if (libc_errno == EINTR)
        continue;
      // Error detected, return AT_NULL
      buf.id = AT_NULL;
      buf.value = AT_NULL;
      break;
    }
    ptr += ret;
    size -= ret;
  }
  return buf;
}

LLVM_LIBC_FUNCTION(unsigned long, getauxval, (unsigned long id)) {
  // Fast path when libc is loaded by its own initialization code. In this case,
  // app.auxv_ptr is already set to the auxv passed on the initial stack of the
  // process.
  AuxvErrnoGuard errno_guard;

  auto search_auxv = [&errno_guard](AuxEntry *auxv,
                                    unsigned long id) -> AuxEntryType {
    for (auto *ptr = auxv; ptr->id != AT_NULL; ptr++)
      if (ptr->id == id)
        return ptr->value;

    errno_guard.mark_failure();
    return AT_NULL;
  };

  // App is a weak symbol that is only defined if libc is linked to its own
  // initialization routine. We need to check if it is null.
  if (&app != nullptr)
    return search_auxv(app.auxv_ptr, id);

  static FutexWordType once_flag;
  LIBC_NAMESPACE::callonce(reinterpret_cast<CallOnceFlag *>(&once_flag),
                           initialize_auxv_once);
  if (auxv != nullptr)
    return search_auxv(auxv, id);

  // Fallback to use read without mmap
  AuxvFdGuard fd_guard;
  if (fd_guard.valid()) {
    while (true) {
      AuxEntry buf = read_entry(fd_guard.get());
      if (buf.id == AT_NULL)
        break;
      if (buf.id == id)
        return buf.value;
    }
  }

  // cannot find the entry after all methods, mark failure and return 0
  errno_guard.mark_failure();
  return AT_NULL;
}
} // namespace LIBC_NAMESPACE_DECL
