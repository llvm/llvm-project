//===-- Named semaphore implementation for Linux --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/semaphore/linux/semaphore.h"

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/ftruncate.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getrandom.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/link.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/mmap.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/munmap.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/unlink.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/sys/mman/linux/shm_common.h"

#include <linux/mman.h> // PROT_READ, PROT_WRITE, MAP_SHARED

namespace LIBC_NAMESPACE_DECL {

namespace {

// define SEM_VALUE_MAX as INT_MAX
constexpr unsigned int SEM_VALUE_MAX =
    static_cast<unsigned int>(cpp::numeric_limits<int>::max());

// Named semaphores are backed by files in /dev/shm/.
// a prefix "sem." is added to avoid name collision.
constexpr cpp::string_view SEM_PREFIX = "/dev/shm/sem.";

// use temporary file to solve data race and guarantee atomic publish.
// Temporary file use different prefix.
constexpr cpp::string_view SEM_TMP_PREFIX = "/dev/shm/sem.tmp_";

// 8 random bytes from getrandom() produce a 16 character hex suffix, giving
// 2^64 possible temp names to avoid collision.
constexpr size_t RANDOM_SUFFIX_BYTES = 8;
constexpr size_t RANDOM_SUFFIX_HEX_LEN = RANDOM_SUFFIX_BYTES * 2;

// fixed-size buffer for the temp path.
using TmpPath =
    cpp::array<char, SEM_TMP_PREFIX.size() + RANDOM_SUFFIX_HEX_LEN + 1>;

// O_NOFOLLOW prevents symlink attacks to /dev/shm/. O_CLOEXEC ensures the
// fd is not leaked to child processes across exec.
constexpr int DEFAULT_OFLAGS = O_NOFOLLOW | O_CLOEXEC;

ErrorOr<TmpPath> generate_tmp_path() {
  // fill out 8 random bytes.
  cpp::array<uint8_t, RANDOM_SUFFIX_BYTES> rand_bytes;
  auto ret = linux_syscalls::getrandom(rand_bytes.data(), rand_bytes.size(), 0);
  if (!ret.has_value())
    return Error(ret.error());

  TmpPath path;
  inline_memcpy(path.data(), SEM_TMP_PREFIX.data(), SEM_TMP_PREFIX.size());

  // Encode each random byte as two hex digits, and fill out tmp path.
  char *dst = path.data() + SEM_TMP_PREFIX.size();
  for (size_t i = 0; i < RANDOM_SUFFIX_BYTES; ++i) {
    *dst++ = internal::int_to_b36_char(rand_bytes[i] >> 4);
    *dst++ = internal::int_to_b36_char(rand_bytes[i] & 0xf);
  }
  *dst = '\0';
  return path;
}

// map an open semaphore fd into memory.
ErrorOr<Semaphore *> map_semaphore(int fd) {
  auto mmap_or = linux_syscalls::mmap(
      nullptr, sizeof(Semaphore), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  linux_syscalls::close(fd);

  if (!mmap_or.has_value())
    return Error(mmap_or.error());

  return reinterpret_cast<Semaphore *>(mmap_or.value());
}

// open an existing named semaphore file and map it.
ErrorOr<Semaphore *> open_existing(const char *path) {
  auto fd_or = linux_syscalls::open(path, O_RDWR | DEFAULT_OFLAGS, 0);
  if (!fd_or.has_value())
    return Error(fd_or.error());
  return map_semaphore(fd_or.value());
}

} // anonymous namespace

ErrorOr<Semaphore *> Semaphore::open(const char *name, int oflag, mode_t mode,
                                     unsigned int value) {
  auto path_or = shm_common::translate_name<SEM_PREFIX>(name);
  if (!path_or.has_value())
    return Error(path_or.error());

  // open an existing semaphore.
  if (!(oflag & O_CREAT))
    return open_existing(path_or->data());

  // check semaphore value.
  if (value > SEM_VALUE_MAX)
    return Error(EINVAL);

  // two step creation:
  // 1. create and fully initialize a temporary file.
  // 2. link() it and publish to the final path atomically.
  // This ensures no other process can observe a partially-initialized
  // semaphore through the final path. If link() fails with EEXIST and
  // O_EXCL is not set, fall back to opening the existing semaphore.

  auto tmp_or = generate_tmp_path();
  if (!tmp_or.has_value())
    return Error(tmp_or.error());

  // if two process happen to map the same random tmp_path, though rare
  // in 2^64 namespace, one succees and the other return EEXIST.
  auto fd_or = linux_syscalls::open(
      tmp_or->data(), O_RDWR | O_CREAT | O_EXCL | DEFAULT_OFLAGS, mode);
  if (!fd_or.has_value())
    return Error(fd_or.error());

  int fd = fd_or.value();

  // resizing temporary semaphore backing file.
  auto trunc_or =
      linux_syscalls::ftruncate(fd, static_cast<off_t>(sizeof(Semaphore)));
  if (!trunc_or.has_value()) {
    linux_syscalls::close(fd);
    linux_syscalls::unlink(tmp_or->data());
    return Error(trunc_or.error());
  }

  // map_semaphore closes the fd.
  auto sem_or = map_semaphore(fd);
  if (!sem_or.has_value()) {
    linux_syscalls::unlink(tmp_or->data());
    return Error(sem_or.error());
  }

  Semaphore *sem = sem_or.value();
  new (sem) Semaphore(value);

  // atomically publish the fully initialized semaphore.
  auto link_or = linux_syscalls::link(tmp_or->data(), path_or->data());

  // temp file is no longer needed.
  linux_syscalls::unlink(tmp_or->data());

  // link() succees
  if (link_or.has_value())
    return sem;

  // link() fail, clean up the mapping.
  linux_syscalls::munmap(sem, sizeof(Semaphore));

  // if the name already exists and O_EXCL was not set, open existing.
  if (link_or.error() == EEXIST && !(oflag & O_EXCL))
    return open_existing(path_or->data());

  return Error(link_or.error());
}

int Semaphore::close(Semaphore *sem) {
  auto result = linux_syscalls::munmap(sem, sizeof(Semaphore));
  if (!result.has_value())
    return result.error();
  return 0;
}

int Semaphore::unlink(const char *name) {
  auto path_or = shm_common::translate_name<SEM_PREFIX>(name);
  if (!path_or.has_value())
    return path_or.error();

  auto result = linux_syscalls::unlink(path_or->data());
  if (!result.has_value())
    return result.error();
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
