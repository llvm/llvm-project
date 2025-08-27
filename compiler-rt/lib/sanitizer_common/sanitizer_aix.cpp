//===-- sanitizer_aix.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries and
// implements AIX-specific functions.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_AIX
#  include <dlfcn.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <pthread.h>
#  include <sched.h>
#  include <signal.h>
#  include <stdio.h>
#  include <stdlib.h>
#  include <string.h>
#  include <sys/errno.h>
#  include <sys/mman.h>
#  include <sys/procfs.h>
#  include <sys/stat.h>
#  include <sys/thread.h>
#  include <sys/time.h>
#  include <sys/types.h>
#  include <sys/ucontext.h>
#  include <unistd.h>

#  include "interception/interception.h"
#  include "sanitizer_aix.h"
#  include "sanitizer_common.h"
#  include "sanitizer_file.h"
#  include "sanitizer_libc.h"
#  include "sanitizer_procmaps.h"

extern char **environ;
extern char **p_xargv;

namespace __sanitizer {

#  include "sanitizer_syscall_generic.inc"

static void *GetFuncAddr(const char *name) {
  // FIXME: if we are going to ship dynamic asan library, we may need to search
  // all the loaded modules with RTLD_DEFAULT if RTLD_NEXT failed.
  void *addr = dlsym(RTLD_NEXT, name);
  return addr;
}

// Internal implementation for the libc functions are also calling to the same
// name function in libc. However because the same name function to libc may be
// intercepted, and in the interceptor function, it may call REAL(func). But the
// REAL(func) may be not assigned at this time, because internal_func may be
// called before interceptor init functions are called. So we need to call to
// libc function via function pointer.

#  define _REAL(func, ...) real##_##func(__VA_ARGS__)

#  define DEFINE__REAL(ret_type, func, ...)                       \
    static ret_type (*real_##func)(__VA_ARGS__) = NULL;           \
    if (!real_##func) {                                           \
      real_##func = (ret_type(*)(__VA_ARGS__))GetFuncAddr(#func); \
    }                                                             \
    CHECK(real_##func);

uptr internal_mmap(void *addr, uptr length, int prot, int flags, int fd,
                   u64 offset) {
  DEFINE__REAL(uptr, mmap, void *addr, uptr length, int prot, int flags, int fd,
               u64 offset);
  return _REAL(mmap, addr, length, prot, flags, fd, offset);
}

uptr internal_munmap(void *addr, uptr length) {
  DEFINE__REAL(uptr, munmap, void *addr, uptr length);
  return _REAL(munmap, addr, length);
}

int internal_mprotect(void *addr, uptr length, int prot) {
  DEFINE__REAL(int, mprotect, void *addr, uptr length, int prot);
  return _REAL(mprotect, addr, length, prot);
}

int internal_madvise(uptr addr, uptr length, int advice) {
  char *raddr = reinterpret_cast<char *>(addr);
  DEFINE__REAL(int, madvise, char *raddr, uptr length, int advice)
  return _REAL(madvise, raddr, length, advice);
}

uptr internal_close(fd_t fd) {
  DEFINE__REAL(uptr, close, fd_t fd);
  return _REAL(close, fd);
}

uptr internal_open(const char *filename, int flags) {
  DEFINE__REAL(uptr, open, const char *filename, int flags);
  return _REAL(open, filename, flags);
}

uptr internal_open(const char *filename, int flags, u32 mode) {
  DEFINE__REAL(uptr, open, const char *filename, int flags, u32 mode);
  return _REAL(open, filename, flags, mode);
}

__sanitizer_FILE *internal_popen(const char *command, const char *type) {
  DEFINE__REAL(__sanitizer_FILE *, popen, const char *command,
               const char *type);
  return _REAL(popen, command, type);
}

int internal_pclose(__sanitizer_FILE *file) {
  FILE *rfile = reinterpret_cast<FILE *>(file);
  DEFINE__REAL(int, pclose, FILE *file);
  return _REAL(pclose, rfile);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  DEFINE__REAL(uptr, read, fd_t fd, void *buf, uptr count);
  return _REAL(read, fd, buf, count);
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  DEFINE__REAL(uptr, write, fd_t fd, const void *buf, uptr count);
  return _REAL(write, fd, buf, count);
}

uptr internal_stat(const char *path, void *buf) {
  DEFINE__REAL(uptr, stat, const char *path, void *buf);
  return _REAL(stat, path, buf);
}

uptr internal_lstat(const char *path, void *buf) {
  DEFINE__REAL(uptr, lstat, const char *path, void *buf);
  return _REAL(lstat, path, buf);
}

uptr internal_fstat(fd_t fd, void *buf) {
  DEFINE__REAL(uptr, fstat, fd_t fd, void *buf);
  return _REAL(fstat, fd, buf);
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

uptr internal_dup(int oldfd) {
  DEFINE__REAL(uptr, dup, int oldfd);
  return _REAL(dup, oldfd);
}

uptr internal_dup2(int oldfd, int newfd) {
  DEFINE__REAL(uptr, dup2, int oldfd, int newfd);
  return _REAL(dup2, oldfd, newfd);
}

uptr internal_readlink(const char *path, char *buf, uptr bufsize) {
  DEFINE__REAL(uptr, readlink, const char *path, char *buf, uptr bufsize);
  return _REAL(readlink, path, buf, bufsize);
}

uptr internal_unlink(const char *path) {
  DEFINE__REAL(uptr, unlink, const char *path);
  return _REAL(unlink, path);
}

uptr internal_sched_yield() {
  DEFINE__REAL(uptr, sched_yield);
  return _REAL(sched_yield);
}

void FutexWait(atomic_uint32_t *p, u32 cmp) { internal_sched_yield(); }

void FutexWake(atomic_uint32_t *p, u32 count) {}

void internal__exit(int exitcode) {
  DEFINE__REAL(void, _exit, int exitcode);
  _REAL(_exit, exitcode);
  Die();  // Unreachable.
}

void internal_usleep(u64 useconds) {
  DEFINE__REAL(void, usleep, u64 useconds);
  _REAL(usleep, useconds);
}

uptr internal_getpid() {
  DEFINE__REAL(uptr, getpid);
  return _REAL(getpid);
}

int internal_dlinfo(void *handle, int request, void *p) { return 0; }

int internal_sigaction(int signum, const void *act, void *oldact) {
  DEFINE__REAL(int, sigaction, int signum, const void *act, void *oldact);
  return _REAL(sigaction, signum, act, oldact);
}

void internal_sigfillset(__sanitizer_sigset_t *set) {
  sigset_t *rset = reinterpret_cast<sigset_t *>(set);
  DEFINE__REAL(void, sigfillset, sigset_t *rset);
  _REAL(sigfillset, rset);
}

uptr internal_sigprocmask(int how, __sanitizer_sigset_t *set,
                          __sanitizer_sigset_t *oldset) {
  sigset_t *rset = reinterpret_cast<sigset_t *>(set);
  sigset_t *roldset = reinterpret_cast<sigset_t *>(oldset);
  DEFINE__REAL(uptr, sigprocmask, int how, sigset_t *rset, sigset_t *roldset);
  return _REAL(sigprocmask, how, rset, roldset);
}

char *internal_getcwd(char *buf, uptr size) {
  DEFINE__REAL(char *, getcwd, char *buf, uptr size);
  return _REAL(getcwd, buf, size);
}

int internal_fork() {
  DEFINE__REAL(int, fork);
  return _REAL(fork);
}

uptr internal_execve(const char *filename, char *const argv[],
                     char *const envp[]) {
  DEFINE__REAL(uptr, execve, const char *filename, char *const argv[],
               char *const envp[]);
  return _REAL(execve, filename, argv, envp);
}

uptr internal_waitpid(int pid, int *status, int options) {
  DEFINE__REAL(uptr, waitpid, int pid, int *status, int options);
  return _REAL(waitpid, pid, status, options);
}

int internal_pthread_join(pthread_t thread, void **status) {
  DEFINE__REAL(int, pthread_join, pthread_t thread, void **status);
  return _REAL(pthread_join, thread, status);
}

int internal_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                            void *(*start_routine)(void *), void *arg) {
  DEFINE__REAL(int, pthread_create, pthread_t *thread,
               const pthread_attr_t *attr, void *(*start_routine)(void *),
               void *arg);
  return _REAL(pthread_create, thread, attr, start_routine, arg);
}

void *internal_start_thread(void *(*func)(void *arg), void *arg) {
  // Start the thread with signals blocked, otherwise it can steal user signals.
  __sanitizer_sigset_t set, old;
  internal_sigfillset(&set);
  internal_sigprocmask(SIG_SETMASK, &set, &old);
  pthread_t th;
  internal_pthread_create(&th, 0, func, arg);
  internal_sigprocmask(SIG_SETMASK, &old, 0);
  // pthread_t is unsinged int on AIX
  return reinterpret_cast<void *>(th);
}

void internal_join_thread(void *th) {
  internal_pthread_join((pthread_t)(reinterpret_cast<uptr>(th)), nullptr);
}

uptr internal_clock_gettime(__sanitizer_clockid_t clk_id, void *tp) {
  clock_t rclk_id = reinterpret_cast<clock_t>(clk_id);
  struct timespec *rtp = reinterpret_cast<struct timespec *>(tp);
  DEFINE__REAL(uptr, clock_gettime, clock_t rclk_id, struct timespec *rtp);
  return _REAL(clock_gettime, rclk_id, rtp);
}

void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  CHECK(stack_top);
  CHECK(stack_bottom);
  if (at_initialization) {
    // This is the main thread. Libpthread may not be initialized yet.
    struct rlimit rl;
    CHECK_EQ(getrlimit(RLIMIT_STACK, &rl), 0);

    // Find the mapping that contains a stack variable.
    MemoryMappingLayout proc_maps(/*cache_enabled*/ true);
    if (proc_maps.Error()) {
      *stack_top = *stack_bottom = 0;
      return;
    }
    MemoryMappedSegment segment;
    uptr prev_end = 0;
    while (proc_maps.Next(&segment)) {
      if ((uptr)&rl < segment.end)
        break;
      prev_end = segment.end;
    }

    CHECK((uptr)&rl >= segment.start && (uptr)&rl < segment.end);

    // Get stacksize from rlimit, but clip it so that it does not overlap
    // with other mappings.
    uptr stacksize = rl.rlim_cur;
    if (stacksize > segment.end - prev_end)
      stacksize = segment.end - prev_end;
    // When running with unlimited stack size, we still want to set some limit.
    // The unlimited stack size is caused by 'ulimit -s unlimited'.
    // Also, for some reason, GNU make spawns subprocesses with unlimited stack.
    if (stacksize > kMaxThreadStackSize)
      stacksize = kMaxThreadStackSize;
    *stack_top = segment.end;
    *stack_bottom = segment.end - stacksize;
    return;
  }
  uptr stacksize = 0;
  void *stackaddr = nullptr;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  CHECK_EQ(pthread_getattr_np(pthread_self(), &attr), 0);
  internal_pthread_attr_getstack(&attr, &stackaddr, &stacksize);
  pthread_attr_destroy(&attr);

  *stack_top = (uptr)stackaddr;
  *stack_bottom = (uptr)stackaddr - stacksize;
}

void GetThreadStackAndTls(bool main, uptr *stk_begin, uptr *stk_end,
                          uptr *tls_begin, uptr *tls_end) {
  // FIXME: handle TLS related flag
  *tls_begin = 0;
  *tls_end = 0;

  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_begin = stack_bottom;
  *stk_end = stack_top;
}

const char *GetEnv(const char *name) { return getenv(name); }

tid_t GetTid() { return thread_self(); }

uptr ReadBinaryName(char *buf, uptr buf_len) {
  struct stat statData;
  struct psinfo psinfoData;

  char FilePsinfo[100] = {};
  internal_snprintf(FilePsinfo, 100, "/proc/%d/psinfo", internal_getpid());
  CHECK_EQ(internal_stat(FilePsinfo, &statData), 0);

  const int fd = internal_open(FilePsinfo, O_RDONLY);
  ssize_t readNum = internal_read(fd, &psinfoData, sizeof(psinfoData));
  CHECK_GE(readNum, 0);

  internal_close(fd);
  char *binary_name = (reinterpret_cast<char ***>(psinfoData.pr_argv))[0][0];

  // This is an absulate path.
  if (binary_name[0] == '/')
    return internal_snprintf(buf, buf_len, "%s", binary_name);

  // This is a relative path to the binary, starts with ./ or ../
  if (binary_name[0] == '.') {
    char *path = nullptr;
    if ((path = internal_getcwd(buf, buf_len)) != nullptr)
      return internal_snprintf(buf + internal_strlen(path),
                               buf_len - internal_strlen(path), "/%s",
                               binary_name) +
             internal_strlen(path);
  }

  // This is running a raw binary in the dir where it is from.
  char *path = nullptr;
  if ((path = internal_getcwd(buf, buf_len)) != nullptr) {
    char fullName[kMaxPathLength] = {};
    internal_snprintf(fullName, kMaxPathLength, "%s/%s", path, binary_name);
    if (FileExists(fullName))
      return internal_snprintf(buf + internal_strlen(path),
                               buf_len - internal_strlen(path), "/%s",
                               binary_name) +
             internal_strlen(path);
  }

  // Find the binary in the env PATH.
  if ((path = FindPathToBinary(binary_name)) != nullptr)
    return internal_snprintf(buf, buf_len, "%s", path);

  return 0;
}

// https://www.ibm.com/docs/en/aix/7.3?topic=concepts-system-memory-allocation-using-malloc-subsystem
uptr GetMaxVirtualAddress() {
#  if SANITIZER_WORDSIZE == 64
  return (1ULL << 60) - 1;
#  else
  return 0xffffffff;
#  endif
}

uptr GetMaxUserVirtualAddress() { return GetMaxVirtualAddress(); }

uptr ReadLongProcessName(/*out*/ char *buf, uptr buf_len) {
  return ReadBinaryName(buf, buf_len);
}

void InitializePlatformCommonFlags(CommonFlags *cf) {}

void InitializePlatformEarly() {
  // Do nothing.
}

uptr GetPageSize() { return getpagesize(); }

void CheckASLR() {
  // Do nothing
}

HandleSignalMode GetHandleSignalMode(int signum) {
  switch (signum) {
    case SIGABRT:
      return common_flags()->handle_abort;
    case SIGILL:
      return common_flags()->handle_sigill;
    case SIGTRAP:
      return common_flags()->handle_sigtrap;
    case SIGFPE:
      return common_flags()->handle_sigfpe;
    case SIGSEGV:
      return common_flags()->handle_segv;
    case SIGBUS:
      return common_flags()->handle_sigbus;
  }
  return kHandleSignalNo;
}

void InitTlsSize() {}

bool FileExists(const char *filename) {
  struct stat st;
  if (internal_stat(filename, &st))
    return false;
  // Sanity check: filename is a regular file.
  return S_ISREG(st.st_mode);
}

bool DirExists(const char *path) {
  struct stat st;
  if (internal_stat(path, &st))
    return false;
  return S_ISDIR(st.st_mode);
}

uptr GetTlsSize() {
  // FIXME: implement this interface.
  return 0;
}

SignalContext::WriteFlag SignalContext::GetWriteFlag() const {
  return SignalContext::Unknown;
}

bool SignalContext::IsTrueFaultingAddress() const { return true; }

void SignalContext::InitPcSpBp() {
  ucontext_t *ucontext = (ucontext_t *)context;
  pc = ucontext->uc_mcontext.jmp_context.iar;
  sp = ucontext->uc_mcontext.jmp_context.gpr[1];
  // The powerpc{,64} ABIs do not specify r31 as the frame pointer, but compiler
  // always uses r31 when we need a frame pointer.
  bp = ucontext->uc_mcontext.jmp_context.gpr[31];
}

void SignalContext::DumpAllRegisters(void *context) {}

char **GetEnviron() { return environ; }

char **GetArgv() { return p_xargv; }

void ListOfModules::init() {
  clearOrInit();
  MemoryMappingLayout memory_mapping(false);
  memory_mapping.DumpListOfModules(&modules_);
}

void ListOfModules::fallbackInit() { clear(); }

u64 MonotonicNanoTime() {
  timespec ts;
  internal_clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * (1000ULL * 1000 * 1000) + ts.tv_nsec;
}

// FIXME implement on this platform.
void GetMemoryProfile(fill_profile_f cb, uptr *stats) {}

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
