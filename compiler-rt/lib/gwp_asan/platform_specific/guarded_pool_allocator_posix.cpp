//===-- guarded_pool_allocator_posix.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/guarded_pool_allocator.h"

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef ANDROID
#include <sys/prctl.h>
#define PR_SET_VMA 0x53564d41
#define PR_SET_VMA_ANON_NAME 0
#endif // ANDROID

void MaybeSetMappingName(void *Mapping, size_t Size, const char *Name) {
#ifdef ANDROID
  prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, Mapping, Size, Name);
#endif // ANDROID
  // Anonymous mapping names are only supported on Android.
  return;
}

namespace gwp_asan {
void *GuardedPoolAllocator::mapMemory(size_t Size, const char *Name) const {
  void *Ptr =
      mmap(nullptr, Size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

  if (Ptr == MAP_FAILED) {
    Printf("Failed to map guarded pool allocator memory, errno: %d\n", errno);
    Printf("  mmap(nullptr, %zu, ...) failed.\n", Size);
    exit(EXIT_FAILURE);
  }
  MaybeSetMappingName(Ptr, Size, Name);
  return Ptr;
}

void GuardedPoolAllocator::unmapMemory(void *Ptr, size_t Size,
                                       const char *Name) const {
  int Res = munmap(Ptr, Size);

  if (Res != 0) {
    Printf("Failed to unmap guarded pool allocator memory, errno: %d\n", errno);
    Printf("  unmmap(%p, %zu, ...) failed.\n", Ptr, Size);
    exit(EXIT_FAILURE);
  }
  MaybeSetMappingName(Ptr, Size, Name);
}

void GuardedPoolAllocator::markReadWrite(void *Ptr, size_t Size,
                                         const char *Name) const {
  if (mprotect(Ptr, Size, PROT_READ | PROT_WRITE) != 0) {
    Printf("Failed to set guarded pool allocator memory at as RW, errno: %d\n",
           errno);
    Printf("  mprotect(%p, %zu, RW) failed.\n", Ptr, Size);
    exit(EXIT_FAILURE);
  }
  MaybeSetMappingName(Ptr, Size, Name);
}

void GuardedPoolAllocator::markInaccessible(void *Ptr, size_t Size,
                                            const char *Name) const {
  // mmap() a PROT_NONE page over the address to release it to the system, if
  // we used mprotect() here the system would count pages in the quarantine
  // against the RSS.
  if (mmap(Ptr, Size, PROT_NONE, MAP_FIXED | MAP_ANONYMOUS | MAP_PRIVATE, -1,
           0) == MAP_FAILED) {
    Printf("Failed to set guarded pool allocator memory as inaccessible, "
           "errno: %d\n",
           errno);
    Printf("  mmap(%p, %zu, NONE, ...) failed.\n", Ptr, Size);
    exit(EXIT_FAILURE);
  }
  MaybeSetMappingName(Ptr, Size, Name);
}

size_t GuardedPoolAllocator::getPlatformPageSize() {
  return sysconf(_SC_PAGESIZE);
}

struct sigaction PreviousHandler;
bool SignalHandlerInstalled;

static void sigSegvHandler(int sig, siginfo_t *info, void *ucontext) {
  gwp_asan::GuardedPoolAllocator::reportError(
      reinterpret_cast<uintptr_t>(info->si_addr));

  // Process any previous handlers.
  if (PreviousHandler.sa_flags & SA_SIGINFO) {
    PreviousHandler.sa_sigaction(sig, info, ucontext);
  } else if (PreviousHandler.sa_handler == SIG_IGN ||
             PreviousHandler.sa_handler == SIG_DFL) {
    // If the previous handler was the default handler, or was ignoring this
    // signal, install the default handler and re-raise the signal in order to
    // get a core dump and terminate this process.
    signal(SIGSEGV, SIG_DFL);
    raise(SIGSEGV);
  } else {
    PreviousHandler.sa_handler(sig);
  }
}

void GuardedPoolAllocator::installAtFork() {
  auto Disable = []() {
    if (auto *S = getSingleton())
      S->disable();
  };
  auto Enable = []() {
    if (auto *S = getSingleton())
      S->enable();
  };
  pthread_atfork(Disable, Enable, Enable);
}

void GuardedPoolAllocator::installSignalHandlers() {
  struct sigaction Action;
  Action.sa_sigaction = sigSegvHandler;
  Action.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &Action, &PreviousHandler);
  SignalHandlerInstalled = true;
}

void GuardedPoolAllocator::uninstallSignalHandlers() {
  if (SignalHandlerInstalled) {
    sigaction(SIGSEGV, &PreviousHandler, nullptr);
    SignalHandlerInstalled = false;
  }
}

uint64_t GuardedPoolAllocator::getThreadID() {
#ifdef SYS_gettid
  return syscall(SYS_gettid);
#else
  return kInvalidThreadID;
#endif
}

} // namespace gwp_asan
