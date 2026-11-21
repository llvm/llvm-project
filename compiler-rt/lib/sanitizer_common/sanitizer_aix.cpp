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
// Declare the large-file interfaces, such as mmap64, so that 64-bit file
// offsets are not truncated in 32-bit mode.
#  ifndef _LARGE_FILE_API
#    define _LARGE_FILE_API
#  endif
#  include <dlfcn.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <pthread.h>
#  include <sched.h>
#  include <signal.h>
#  include <sys/mman.h>
#  include <sys/resource.h>
#  include <sys/stat.h>
#  include <sys/thread.h>
#  include <sys/types.h>
#  include <sys/ucontext.h>
#  include <sys/wait.h>
#  include <time.h>
#  include <unistd.h>

#  include "sanitizer_aix.h"
#  include "sanitizer_common.h"
#  include "sanitizer_file.h"
#  include "sanitizer_libc.h"
#  include "sanitizer_procmaps.h"

extern char** environ;
extern char** p_xargv;

namespace __sanitizer {

#  include "sanitizer_syscall_generic.inc"

static void* GetFuncAddr(const char* name) {
  // FIXME: if we are going to ship dynamic asan library, we may need to search
  // all the loaded modules with RTLD_DEFAULT if RTLD_NEXT failed.
  void* addr = dlsym(RTLD_NEXT, name);
  return addr;
}

// Internal implementation for the libc functions are also calling to the same
// name function in libc. However because the same name function to libc may be
// intercepted, and in the interceptor function, it may call REAL(func). But the
// REAL(func) may be not assigned at this time, because internal_func may be
// called before interceptor init functions are called. So we need to call to
// libc function via function pointer.
//
// The function pointer has the type declared by the libc headers. If the
// headers map a function to another symbol (as they do when _LARGE_FILES is
// defined), the symbol is looked up by the name that `func` expands to.

#  define _REAL(func, ...) real_##func(__VA_ARGS__)

#  define DEFINE__REAL(func)                             \
    static decltype(&::func) real_##func = nullptr;      \
    if (!real_##func) {                                  \
      real_##func = reinterpret_cast<decltype(&::func)>( \
          GetFuncAddr(SANITIZER_STRINGIFY(func)));       \
    }                                                    \
    CHECK(real_##func)

uptr internal_mmap(void* addr, uptr length, int prot, int flags, int fd,
                   u64 offset) {
  DEFINE__REAL(mmap64);
  return reinterpret_cast<uptr>(
      _REAL(mmap64, addr, length, prot, flags, fd, offset));
}

uptr internal_munmap(void* addr, uptr length) {
  DEFINE__REAL(munmap);
  return _REAL(munmap, addr, length);
}

int internal_mprotect(void* addr, uptr length, int prot) {
  DEFINE__REAL(mprotect);
  return _REAL(mprotect, addr, length, prot);
}

int internal_madvise(uptr addr, uptr length, int advice) {
  DEFINE__REAL(madvise);
  return _REAL(madvise, reinterpret_cast<caddr_t>(addr), length, advice);
}

uptr internal_close_range(fd_t lowfd, fd_t highfd, int flags) {
  return -1;  // Not supported.
}

uptr internal_close(fd_t fd) {
  DEFINE__REAL(close);
  return _REAL(close, fd);
}

uptr internal_open(const char* filename, int flags) {
  DEFINE__REAL(open);
  return _REAL(open, filename, flags);
}

uptr internal_open(const char* filename, int flags, u32 mode) {
  DEFINE__REAL(open);
  return _REAL(open, filename, flags, mode);
}

uptr internal_read(fd_t fd, void* buf, uptr count) {
  DEFINE__REAL(read);
  return _REAL(read, fd, buf, count);
}

uptr internal_write(fd_t fd, const void* buf, uptr count) {
  DEFINE__REAL(write);
  return _REAL(write, fd, buf, count);
}

uptr internal_stat(const char* path, void* buf) {
  DEFINE__REAL(stat);
  return _REAL(stat, path, static_cast<struct stat*>(buf));
}

uptr internal_lstat(const char* path, void* buf) {
  DEFINE__REAL(lstat);
  return _REAL(lstat, path, static_cast<struct stat*>(buf));
}

uptr internal_fstat(fd_t fd, void* buf) {
  DEFINE__REAL(fstat);
  return _REAL(fstat, fd, static_cast<struct stat*>(buf));
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

uptr internal_dup(int oldfd) {
  DEFINE__REAL(dup);
  return _REAL(dup, oldfd);
}

uptr internal_dup2(int oldfd, int newfd) {
  DEFINE__REAL(dup2);
  return _REAL(dup2, oldfd, newfd);
}

uptr internal_readlink(const char* path, char* buf, uptr bufsize) {
  DEFINE__REAL(readlink);
  return _REAL(readlink, path, buf, bufsize);
}

uptr internal_unlink(const char* path) {
  DEFINE__REAL(unlink);
  return _REAL(unlink, path);
}

uptr internal_sched_yield() {
  DEFINE__REAL(sched_yield);
  return _REAL(sched_yield);
}

void FutexWait(atomic_uint32_t* p, u32 cmp) {
  // FIXME: implement actual blocking.
  internal_sched_yield();
}

void FutexWake(atomic_uint32_t* p, u32 count) {}

void internal__exit(int exitcode) {
  DEFINE__REAL(_exit);
  _REAL(_exit, exitcode);
  Die();  // Unreachable.
}

void internal_usleep(u64 useconds) {
  struct timespec ts;
  ts.tv_sec = useconds / 1000000;
  ts.tv_nsec = (useconds % 1000000) * 1000;
  DEFINE__REAL(nanosleep);
  _REAL(nanosleep, &ts, &ts);
}

uptr internal_getpid() {
  DEFINE__REAL(getpid);
  return _REAL(getpid);
}

static char* internal_getcwd(char* buf, uptr size) {
  DEFINE__REAL(getcwd);
  return _REAL(getcwd, buf, size);
}

int internal_sigaction(int signum, const void* act, void* oldact) {
  DEFINE__REAL(sigaction);
  return _REAL(sigaction, signum, static_cast<const struct sigaction*>(act),
               static_cast<struct sigaction*>(oldact));
}

void internal_sigfillset(__sanitizer_sigset_t* set) {
  DEFINE__REAL(sigfillset);
  _REAL(sigfillset, reinterpret_cast<sigset_t*>(set));
}

uptr internal_sigprocmask(int how, __sanitizer_sigset_t* set,
                          __sanitizer_sigset_t* oldset) {
  DEFINE__REAL(sigprocmask);
  return _REAL(sigprocmask, how, reinterpret_cast<sigset_t*>(set),
               reinterpret_cast<sigset_t*>(oldset));
}

int internal_fork() {
  DEFINE__REAL(fork);
  return _REAL(fork);
}

uptr internal_execve(const char* filename, char* const argv[],
                     char* const envp[]) {
  DEFINE__REAL(execve);
  return _REAL(execve, filename, argv, envp);
}

uptr internal_waitpid(int pid, int* status, int options) {
  DEFINE__REAL(waitpid);
  return _REAL(waitpid, pid, status, options);
}

void* internal_start_thread(void* (*func)(void* arg), void* arg) {
  if (&internal_pthread_create == 0)
    return nullptr;
  // Start the thread with signals blocked, otherwise it can steal user signals.
  __sanitizer_sigset_t set, old;
  internal_sigfillset(&set);
  internal_sigprocmask(SIG_SETMASK, &set, &old);
  // pthread_t is an unsigned int on AIX, which is narrower than void* in 64-bit
  // mode, so it cannot be stored to a void* directly.
  pthread_t th;
  internal_pthread_create(&th, nullptr, func, arg);
  internal_sigprocmask(SIG_SETMASK, &old, nullptr);
  return reinterpret_cast<void*>(static_cast<uptr>(th));
}

void internal_join_thread(void* th) {
  if (&internal_pthread_join)
    internal_pthread_join(th, nullptr);
}

uptr internal_clock_gettime(__sanitizer_clockid_t clk_id, void* tp) {
  DEFINE__REAL(clock_gettime);
  return _REAL(clock_gettime, static_cast<clockid_t>(clk_id),
               static_cast<struct timespec*>(tp));
}

const char* GetEnv(const char* name) {
  if (!environ)
    return nullptr;
  uptr name_len = internal_strlen(name);
  for (char** env = environ; *env; env++) {
    if (internal_strncmp(*env, name, name_len) == 0 && (*env)[name_len] == '=')
      return *env + name_len + 1;
  }
  return nullptr;
}

ThreadID GetTid() { return thread_self(); }

// The 64-bit AIX user address space ends with the stack segment at
// 0x0fffffffffffffff. See
// https://www.ibm.com/docs/en/aix/7.3?topic=concepts-system-memory-allocation-using-malloc-subsystem
uptr GetMaxVirtualAddress() {
#  if SANITIZER_WORDSIZE == 64
  return (1ULL << 60) - 1;
#  else
  return 0xffffffff;
#  endif
}

uptr GetMaxUserVirtualAddress() { return GetMaxVirtualAddress(); }

void GetThreadStackTopAndBottom(bool at_initialization, uptr* stack_top,
                                uptr* stack_bottom) {
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
    if (stacksize > kMaxThreadStackSize)
      stacksize = kMaxThreadStackSize;
    *stack_top = segment.end;
    *stack_bottom = segment.end - stacksize;

    // The stack segment ends at the end of the address space, so its end is
    // one past the maximum address. Keep stack_top in range so that shadow
    // address calculations are valid.
    uptr max_addr = GetMaxUserVirtualAddress();
    if (*stack_top == max_addr + 1)
      *stack_top = max_addr;
    else
      CHECK_LE(*stack_top, max_addr);
    return;
  }
  uptr stacksize = 0;
  void* stackaddr = nullptr;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  CHECK_EQ(pthread_getattr_np(pthread_self(), &attr), 0);
  internal_pthread_attr_getstack(&attr, &stackaddr, &stacksize);
  pthread_attr_destroy(&attr);

  *stack_top = (uptr)stackaddr;
  *stack_bottom = (uptr)stackaddr - stacksize;
}

void GetThreadStackAndTls(bool main, uptr* stk_begin, uptr* stk_end,
                          uptr* tls_begin, uptr* tls_end) {
  // FIXME: implement TLS bounds.
  *tls_begin = 0;
  *tls_end = 0;

  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_begin = stack_bottom;
  *stk_end = stack_top;
}

uptr ReadBinaryName(/*out*/ char* buf, uptr buf_len) {
  if (buf_len == 0)
    return 0;
  buf[0] = '\0';
  char** argv = GetArgv();
  if (!argv || !argv[0])
    return 0;

  // AIX has no /proc/self/exe, so derive the path from argv[0]. It is either a
  // path, which may be relative to the current directory, or a name found in
  // PATH. The symbolizer needs an absolute path.
  char* path = FindPathToBinary(argv[0]);
  if (!path)
    return 0;
  uptr len = 0;
  if (path[0] == '/') {
    len = internal_snprintf(buf, buf_len, "%s", path);
  } else if (internal_getcwd(buf, buf_len)) {
    uptr cwd_len = internal_strlen(buf);
    len = cwd_len +
          internal_snprintf(buf + cwd_len, buf_len - cwd_len, "/%s", path);
  } else {
    buf[0] = '\0';
  }
  InternalFree(path);
  return Min(len, buf_len - 1);
}

uptr ReadLongProcessName(/*out*/ char* buf, uptr buf_len) {
  return ReadBinaryName(buf, buf_len);
}

void InitializePlatformCommonFlags(CommonFlags* cf) {}

void InitializePlatformEarly() {}

uptr GetPageSize() { return getpagesize(); }

void CheckASLR() {}

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

bool FileExists(const char* filename) {
  struct stat st;
  if (internal_stat(filename, &st))
    return false;
  // Sanity check: filename is a regular file.
  return S_ISREG(st.st_mode);
}

bool DirExists(const char* path) {
  struct stat st;
  if (internal_stat(path, &st))
    return false;
  return S_ISDIR(st.st_mode);
}

uptr GetTlsSize() {
  // FIXME: implement.
  return 0;
}

SignalContext::WriteFlag SignalContext::GetWriteFlag() const {
  return SignalContext::Unknown;
}

bool SignalContext::IsTrueFaultingAddress() const { return true; }

void SignalContext::InitPcSpBp() {
  ucontext_t* ucontext = static_cast<ucontext_t*>(context);
  pc = ucontext->uc_mcontext.jmp_context.iar;
  sp = ucontext->uc_mcontext.jmp_context.gpr[1];
  // The PowerPC ABIs do not specify a frame pointer, but compilers use r31
  // when one is needed.
  bp = ucontext->uc_mcontext.jmp_context.gpr[31];
}

void SignalContext::DumpAllRegisters(void* context) {}

char** GetEnviron() { return environ; }

// p_xargv is set by the AIX C runtime to the argv passed to main.
char** GetArgv() { return p_xargv; }

void ListOfModules::init() {
  clearOrInit();
  MemoryMappingLayout memory_mapping(/*cache_enabled*/ false);
  memory_mapping.DumpListOfModules(&modules_);
}

void ListOfModules::fallbackInit() { clear(); }

u64 MonotonicNanoTime() {
  timespec ts;
  internal_clock_gettime(CLOCK_MONOTONIC, &ts);
  return (u64)ts.tv_sec * (1000ULL * 1000 * 1000) + ts.tv_nsec;
}

// FIXME: implement.
void GetMemoryProfile(fill_profile_f cb, uptr* stats) {}

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
