//===-- sanitizer_linux.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Linux-specific syscall wrappers and classes.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_LINUX_H
#define SANITIZER_LINUX_H

#include "sanitizer_platform.h"
#if SANITIZER_FREEBSD || SANITIZER_LINUX
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_posix.h"
#include "sanitizer_platform_limits_posix.h"

struct link_map;  // Opaque type returned by dlopen().
struct sigaltstack;

namespace __sanitizer {
// Dirent structure for getdents(). Note that this structure is different from
// the one in <dirent.h>, which is used by readdir().
struct linux_dirent;

// Syscall wrappers.
uptr internal_getdents(fd_t fd, struct linux_dirent *dirp, unsigned int count);
uptr internal_sigaltstack(const struct sigaltstack* ss,
                          struct sigaltstack* oss);
uptr internal_sigprocmask(int how, __sanitizer_sigset_t *set,
    __sanitizer_sigset_t *oldset);

// Linux-only syscalls.
#if SANITIZER_LINUX
uptr internal_prctl(int option, uptr arg2, uptr arg3, uptr arg4, uptr arg5);
// Used only by sanitizer_stoptheworld. Signal handlers that are actually used
// (like the process-wide error reporting SEGV handler) must use
// internal_sigaction instead.
int internal_sigaction_norestorer(int signum, const void *act, void *oldact);
void internal_sigdelset(__sanitizer_sigset_t *set, int signum);
#if defined(__x86_64__) || defined(__mips__) || defined(__aarch64__) \
  || defined(__powerpc64__) || defined(__s390__)
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr);
#endif
#endif  // SANITIZER_LINUX

// This class reads thread IDs from /proc/<pid>/task using only syscalls.
class ThreadLister {
 public:
  explicit ThreadLister(int pid);
  ~ThreadLister();
  // GetNextTID returns -1 if the list of threads is exhausted, or if there has
  // been an error.
  int GetNextTID();
  void Reset();
  bool error();

 private:
  bool GetDirectoryEntries();

  int pid_;
  int descriptor_;
  InternalScopedBuffer<char> buffer_;
  bool error_;
  struct linux_dirent* entry_;
  int bytes_read_;
};

// Exposed for testing.
uptr ThreadDescriptorSize();
uptr ThreadSelf();
uptr ThreadSelfOffset();

// Matches a library's file name against a base name (stripping path and version
// information).
bool LibraryNameIs(const char *full_name, const char *base_name);

// Call cb for each region mapped by map.
void ForEachMappedRegion(link_map *map, void (*cb)(const void *, uptr));
}  // namespace __sanitizer

#endif  // SANITIZER_FREEBSD || SANITIZER_LINUX
#endif  // SANITIZER_LINUX_H
