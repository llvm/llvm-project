//===-- trec_platform_linux.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
// Linux- and BSD-specific code.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_LINUX || SANITIZER_FREEBSD || SANITIZER_NETBSD

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_platform_limits_netbsd.h"
#include "sanitizer_common/sanitizer_platform_limits_posix.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stoptheworld.h"
#include "trec_flags.h"
#include "trec_platform.h"
#include "trec_rtl.h"

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/mman.h>
#if SANITIZER_LINUX
#include <sys/personality.h>
#include <setjmp.h>
#endif
#include <sys/syscall.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sched.h>
#include <dlfcn.h>
#if SANITIZER_LINUX
#define __need_res_state
#include <resolv.h>
#endif

#ifdef sa_handler
# undef sa_handler
#endif

#ifdef sa_sigaction
# undef sa_sigaction
#endif

#if SANITIZER_FREEBSD
extern "C" void *__libc_stack_end;
void *__libc_stack_end = 0;
#endif

#if SANITIZER_LINUX && defined(__aarch64__) && !SANITIZER_GO
# define INIT_LONGJMP_XOR_KEY 1
#else
# define INIT_LONGJMP_XOR_KEY 0
#endif

#if INIT_LONGJMP_XOR_KEY
#include "interception/interception.h"
// Must be declared outside of other namespaces.
DECLARE_REAL(int, _setjmp, void *env)
#endif

namespace __trec {

#if INIT_LONGJMP_XOR_KEY
static void InitializeLongjmpXorKey();
static uptr longjmp_xor_key;
#endif

#ifdef TREC_RUNTIME_VMA
// Runtime detected VMA size.
uptr vmaSize;
#endif

enum {
  MemTotal  = 0,
  MemShadow = 1,
  MemMeta   = 2,
  MemFile   = 3,
  MemMmap   = 4,
  MemTrace  = 5,
  MemHeap   = 6,
  MemOther  = 7,
  MemCount  = 8,
};

void InitializePlatformEarly() {
#ifdef TREC_RUNTIME_VMA
  vmaSize =
    (MostSignificantSetBitIndex(GET_CURRENT_FRAME()) + 1);
#if defined(__aarch64__)
# if !SANITIZER_GO
  if (vmaSize != 39 && vmaSize != 42 && vmaSize != 48) {
    Printf("FATAL: TraceRecorder: unsupported VMA range\n");
    Printf("FATAL: Found %zd - Supported 39, 42 and 48\n", vmaSize);
    Die();
  }
#else
  if (vmaSize != 48) {
    Printf("FATAL: TraceRecorder: unsupported VMA range\n");
    Printf("FATAL: Found %zd - Supported 48\n", vmaSize);
    Die();
  }
#endif
#elif defined(__powerpc64__)
# if !SANITIZER_GO
  if (vmaSize != 44 && vmaSize != 46 && vmaSize != 47) {
    Printf("FATAL: TraceRecorder: unsupported VMA range\n");
    Printf("FATAL: Found %zd - Supported 44, 46, and 47\n", vmaSize);
    Die();
  }
# else
  if (vmaSize != 46 && vmaSize != 47) {
    Printf("FATAL: TraceRecorder: unsupported VMA range\n");
    Printf("FATAL: Found %zd - Supported 46, and 47\n", vmaSize);
    Die();
  }
# endif
#endif
#endif
}

void InitializePlatform() {
  DisableCoreDumperIfNecessary();

  // Go maps shadow memory lazily and works fine with limited address space.
  // Unlimited stack is not a problem as well, because the executable
  // is not compiled with -pie.
#if !SANITIZER_GO
  {
    bool reexec = false;
    // TRec doesn't play well with unlimited stack size (as stack
    // overlaps with shadow memory). If we detect unlimited stack size,
    // we re-exec the program with limited stack size as a best effort.
    if (StackSizeIsUnlimited()) {
      const uptr kMaxStackSize = 32 * 1024 * 1024;
      VReport(1, "Program is run with unlimited stack size, which wouldn't "
                 "work with TraceRecorder.\n"
                 "Re-execing with stack size limited to %zd bytes.\n",
              kMaxStackSize);
      SetStackSizeLimitInBytes(kMaxStackSize);
      reexec = true;
    }

    if (!AddressSpaceIsUnlimited()) {
      Report("WARNING: Program is run with limited virtual address space,"
             " which wouldn't work with TraceRecorder.\n");
      Report("Re-execing with unlimited virtual address space.\n");
      SetAddressSpaceUnlimited();
      reexec = true;
    }
#if SANITIZER_LINUX && defined(__aarch64__)
    // After patch "arm64: mm: support ARCH_MMAP_RND_BITS." is introduced in
    // linux kernel, the random gap between stack and mapped area is increased
    // from 128M to 36G on 39-bit aarch64. As it is almost impossible to cover
    // this big range, we should disable randomized virtual space on aarch64.
    int old_personality = personality(0xffffffff);
    if (old_personality != -1 && (old_personality & ADDR_NO_RANDOMIZE) == 0) {
      VReport(1, "WARNING: Program is run with randomized virtual address "
              "space, which wouldn't work with TraceRecorder.\n"
              "Re-execing with fixed virtual address space.\n");
      CHECK_NE(personality(old_personality | ADDR_NO_RANDOMIZE), -1);
      reexec = true;
    }
    // Initialize the xor key used in {sig}{set,long}jump.
    InitializeLongjmpXorKey();
#endif
    if (reexec)
      ReExec();
  }

  // CheckAndProtect();
  InitTlsSize();
#endif  // !SANITIZER_GO
}

#if !SANITIZER_GO
// Extract file descriptors passed to glibc internal __res_iclose function.
// This is required to properly "close" the fds, because we do not see internal
// closes within glibc. The code is a pure hack.
int ExtractResolvFDs(void *state, int *fds, int nfd) {
#if SANITIZER_LINUX && !SANITIZER_ANDROID
  int cnt = 0;
  struct __res_state *statp = (struct __res_state*)state;
  for (int i = 0; i < MAXNS && cnt < nfd; i++) {
    if (statp->_u._ext.nsaddrs[i] && statp->_u._ext.nssocks[i] != -1)
      fds[cnt++] = statp->_u._ext.nssocks[i];
  }
  return cnt;
#else
  return 0;
#endif
}

// Extract file descriptors passed via UNIX domain sockets.
// This is requried to properly handle "open" of these fds.
// see 'man recvmsg' and 'man 3 cmsg'.
int ExtractRecvmsgFDs(void *msgp, int *fds, int nfd) {
  int res = 0;
  msghdr *msg = (msghdr*)msgp;
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(msg);
  for (; cmsg; cmsg = CMSG_NXTHDR(msg, cmsg)) {
    if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS)
      continue;
    int n = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(fds[0]);
    for (int i = 0; i < n; i++) {
      fds[res++] = ((int*)CMSG_DATA(cmsg))[i];
      if (res == nfd)
        return res;
    }
  }
  return res;
}

#if SANITIZER_NETBSD
# ifdef __x86_64__
#  define LONG_JMP_SP_ENV_SLOT 6
# else
#  error unsupported
# endif
#elif defined(__powerpc__)
# define LONG_JMP_SP_ENV_SLOT 0
#elif SANITIZER_FREEBSD
# define LONG_JMP_SP_ENV_SLOT 2
#elif SANITIZER_LINUX
# ifdef __aarch64__
#  define LONG_JMP_SP_ENV_SLOT 13
# elif defined(__mips64)
#  define LONG_JMP_SP_ENV_SLOT 1
# else
#  define LONG_JMP_SP_ENV_SLOT 6
# endif
#endif

#if INIT_LONGJMP_XOR_KEY
// GLIBC mangles the function pointers in jmp_buf (used in {set,long}*jmp
// functions) by XORing them with a random key.  For AArch64 it is a global
// variable rather than a TCB one (as for x86_64/powerpc).  We obtain the key by
// issuing a setjmp and XORing the SP pointer values to derive the key.
static void InitializeLongjmpXorKey() {
  // 1. Call REAL(setjmp), which stores the mangled SP in env.
  jmp_buf env;
  REAL(_setjmp)(env);

  // 2. Retrieve vanilla/mangled SP.
  uptr sp;
  asm("mov  %0, sp" : "=r" (sp));
  uptr mangled_sp = ((uptr *)&env)[LONG_JMP_SP_ENV_SLOT];

  // 3. xor SPs to obtain key.
  longjmp_xor_key = mangled_sp ^ sp;
}
#endif

void ImitateTlsWrite(ThreadState *thr, uptr tls_addr, uptr tls_size) {
  // Check that the thr object is in tls;
  const uptr thr_beg = (uptr)thr;
  const uptr thr_end = (uptr)thr + sizeof(*thr);
  CHECK_GE(thr_beg, tls_addr);
  CHECK_LE(thr_beg, tls_addr + tls_size);
  CHECK_GE(thr_end, tls_addr);
  CHECK_LE(thr_end, tls_addr + tls_size);
}

// Note: this function runs with async signals enabled,
// so it must not touch any trec state.
int call_pthread_cancel_with_cleanup(int (*fn)(void *arg),
                                     void (*cleanup)(void *arg), void *arg) {
  // pthread_cleanup_push/pop are hardcore macros mess.
  // We can't intercept nor call them w/o including pthread.h.
  int res;
  pthread_cleanup_push(cleanup, arg);
  res = fn(arg);
  pthread_cleanup_pop(0);
  return res;
}
#endif  // !SANITIZER_GO

#if !SANITIZER_GO
void ReplaceSystemMalloc() { }
#endif

#if !SANITIZER_GO
#if SANITIZER_ANDROID
// On Android, one thread can call intercepted functions after
// DestroyThreadState(), so add a fake thread state for "dead" threads.
static ThreadState *dead_thread_state = nullptr;

ThreadState *cur_thread() {
  ThreadState* thr = reinterpret_cast<ThreadState*>(*get_android_tls_ptr());
  if (thr == nullptr) {
    __sanitizer_sigset_t emptyset;
    internal_sigfillset(&emptyset);
    __sanitizer_sigset_t oldset;
    CHECK_EQ(0, internal_sigprocmask(SIG_SETMASK, &emptyset, &oldset));
    thr = reinterpret_cast<ThreadState*>(*get_android_tls_ptr());
    if (thr == nullptr) {
      thr = reinterpret_cast<ThreadState*>(MmapOrDie(sizeof(ThreadState),
                                                     "ThreadState"));
      *get_android_tls_ptr() = reinterpret_cast<uptr>(thr);
      if (dead_thread_state == nullptr) {
        dead_thread_state = reinterpret_cast<ThreadState*>(
            MmapOrDie(sizeof(ThreadState), "ThreadState"));
        dead_thread_state->fast_state.SetIgnoreBit();
        dead_thread_state->ignore_interceptors = 1;
        dead_thread_state->is_dead = true;
        *const_cast<int*>(&dead_thread_state->tid) = -1;
        CHECK_EQ(0, internal_mprotect(dead_thread_state, sizeof(ThreadState),
                                      PROT_READ));
      }
    }
    CHECK_EQ(0, internal_sigprocmask(SIG_SETMASK, &oldset, nullptr));
  }
  return thr;
}

void set_cur_thread(ThreadState *thr) {
  *get_android_tls_ptr() = reinterpret_cast<uptr>(thr);
}

void cur_thread_finalize() {
  __sanitizer_sigset_t emptyset;
  internal_sigfillset(&emptyset);
  __sanitizer_sigset_t oldset;
  CHECK_EQ(0, internal_sigprocmask(SIG_SETMASK, &emptyset, &oldset));
  ThreadState* thr = reinterpret_cast<ThreadState*>(*get_android_tls_ptr());
  if (thr != dead_thread_state) {
    *get_android_tls_ptr() = reinterpret_cast<uptr>(dead_thread_state);
    UnmapOrDie(thr, sizeof(ThreadState));
  }
  CHECK_EQ(0, internal_sigprocmask(SIG_SETMASK, &oldset, nullptr));
}
#endif  // SANITIZER_ANDROID
#endif  // if !SANITIZER_GO

}  // namespace __trec

#endif  // SANITIZER_LINUX || SANITIZER_FREEBSD || SANITIZER_NETBSD
