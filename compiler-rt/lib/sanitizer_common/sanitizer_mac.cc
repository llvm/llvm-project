//===-- sanitizer_mac.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries and
// implements OSX-specific functions.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_MAC
#include "sanitizer_mac.h"

// Use 64-bit inodes in file operations. ASan does not support OS X 10.5, so
// the clients will most certainly use 64-bit ones as well.
#ifndef _DARWIN_USE_64_BIT_INODE
#define _DARWIN_USE_64_BIT_INODE 1
#endif
#include <stdio.h>

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_platform_limits_posix.h"
#include "sanitizer_procmaps.h"

#if !SANITIZER_IOS
#include <crt_externs.h>  // for _NSGetEnviron
#else
extern char **environ;
#endif

#if defined(__has_include) && __has_include(<os/trace.h>)
#define SANITIZER_OS_TRACE 1
#include <os/trace.h>
#else
#define SANITIZER_OS_TRACE 0
#endif

#if !SANITIZER_IOS
#include <crt_externs.h>  // for _NSGetArgv and _NSGetEnviron
#else
extern "C" {
  extern char ***_NSGetArgv(void);
}
#endif

#include <asl.h>
#include <dlfcn.h>  // for dladdr()
#include <errno.h>
#include <fcntl.h>
#include <libkern/OSAtomic.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <util.h>

// from <crt_externs.h>, but we don't have that file on iOS
extern "C" {
  extern char ***_NSGetArgv(void);
  extern char ***_NSGetEnviron(void);
}

namespace __sanitizer {

#include "sanitizer_syscall_generic.inc"

// Direct syscalls, don't call libmalloc hooks.
extern "C" void *__mmap(void *addr, size_t len, int prot, int flags, int fildes,
                        off_t off);
extern "C" int __munmap(void *, size_t);

// ---------------------- sanitizer_libc.h
uptr internal_mmap(void *addr, size_t length, int prot, int flags,
                   int fd, u64 offset) {
  if (fd == -1) fd = VM_MAKE_TAG(VM_MEMORY_ANALYSIS_TOOL);
  return (uptr)__mmap(addr, length, prot, flags, fd, offset);
}

uptr internal_munmap(void *addr, uptr length) {
  return __munmap(addr, length);
}

int internal_mprotect(void *addr, uptr length, int prot) {
  return mprotect(addr, length, prot);
}

uptr internal_close(fd_t fd) {
  return close(fd);
}

uptr internal_open(const char *filename, int flags) {
  return open(filename, flags);
}

uptr internal_open(const char *filename, int flags, u32 mode) {
  return open(filename, flags, mode);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  return read(fd, buf, count);
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  return write(fd, buf, count);
}

uptr internal_stat(const char *path, void *buf) {
  return stat(path, (struct stat *)buf);
}

uptr internal_lstat(const char *path, void *buf) {
  return lstat(path, (struct stat *)buf);
}

uptr internal_fstat(fd_t fd, void *buf) {
  return fstat(fd, (struct stat *)buf);
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

uptr internal_dup2(int oldfd, int newfd) {
  return dup2(oldfd, newfd);
}

uptr internal_readlink(const char *path, char *buf, uptr bufsize) {
  return readlink(path, buf, bufsize);
}

uptr internal_unlink(const char *path) {
  return unlink(path);
}

uptr internal_sched_yield() {
  return sched_yield();
}

void internal__exit(int exitcode) {
  _exit(exitcode);
}

unsigned int internal_sleep(unsigned int seconds) {
  return sleep(seconds);
}

uptr internal_getpid() {
  return getpid();
}

int internal_sigaction(int signum, const void *act, void *oldact) {
  return sigaction(signum,
                   (struct sigaction *)act, (struct sigaction *)oldact);
}

void internal_sigfillset(__sanitizer_sigset_t *set) { sigfillset(set); }

uptr internal_sigprocmask(int how, __sanitizer_sigset_t *set,
                          __sanitizer_sigset_t *oldset) {
  return sigprocmask(how, set, oldset);
}

// Doesn't call pthread_atfork() handlers.
extern "C" pid_t __fork(void);

int internal_fork() {
  return __fork();
}

int internal_forkpty(int *amaster) {
  int master, slave;
  if (openpty(&master, &slave, nullptr, nullptr, nullptr) == -1) return -1;
  int pid = __fork();
  if (pid == -1) {
    close(master);
    close(slave);
    return -1;
  }
  if (pid == 0) {
    close(master);
    if (login_tty(slave) != 0) {
      // We already forked, there's not much we can do.  Let's quit.
      Report("login_tty failed (errno %d)\n", errno);
      internal__exit(1);
    }
  } else {
    *amaster = master;
    close(slave);
  }
  return pid;
}

uptr internal_rename(const char *oldpath, const char *newpath) {
  return rename(oldpath, newpath);
}

uptr internal_ftruncate(fd_t fd, uptr size) {
  return ftruncate(fd, size);
}

uptr internal_execve(const char *filename, char *const argv[],
                     char *const envp[]) {
  return execve(filename, argv, envp);
}

uptr internal_waitpid(int pid, int *status, int options) {
  return waitpid(pid, status, options);
}

// ----------------- sanitizer_common.h
bool FileExists(const char *filename) {
  struct stat st;
  if (stat(filename, &st))
    return false;
  // Sanity check: filename is a regular file.
  return S_ISREG(st.st_mode);
}

uptr GetTid() {
  // FIXME: This can potentially get truncated on 32-bit, where uptr is 4 bytes.
  uint64_t tid;
  pthread_threadid_np(nullptr, &tid);
  return tid;
}

void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  CHECK(stack_top);
  CHECK(stack_bottom);
  uptr stacksize = pthread_get_stacksize_np(pthread_self());
  // pthread_get_stacksize_np() returns an incorrect stack size for the main
  // thread on Mavericks. See
  // https://github.com/google/sanitizers/issues/261
  if ((GetMacosVersion() >= MACOS_VERSION_MAVERICKS) && at_initialization &&
      stacksize == (1 << 19))  {
    struct rlimit rl;
    CHECK_EQ(getrlimit(RLIMIT_STACK, &rl), 0);
    // Most often rl.rlim_cur will be the desired 8M.
    if (rl.rlim_cur < kMaxThreadStackSize) {
      stacksize = rl.rlim_cur;
    } else {
      stacksize = kMaxThreadStackSize;
    }
  }
  void *stackaddr = pthread_get_stackaddr_np(pthread_self());
  *stack_top = (uptr)stackaddr;
  *stack_bottom = *stack_top - stacksize;
}

char **GetEnviron() {
#if !SANITIZER_IOS
  char ***env_ptr = _NSGetEnviron();
  if (!env_ptr) {
    Report("_NSGetEnviron() returned NULL. Please make sure __asan_init() is "
           "called after libSystem_initializer().\n");
    CHECK(env_ptr);
  }
  char **environ = *env_ptr;
#endif
  CHECK(environ);
  return environ;
}

const char *GetEnv(const char *name) {
  char **env = GetEnviron();
  uptr name_len = internal_strlen(name);
  while (*env != 0) {
    uptr len = internal_strlen(*env);
    if (len > name_len) {
      const char *p = *env;
      if (!internal_memcmp(p, name, name_len) &&
          p[name_len] == '=') {  // Match.
        return *env + name_len + 1;  // String starting after =.
      }
    }
    env++;
  }
  return 0;
}

uptr ReadBinaryName(/*out*/char *buf, uptr buf_len) {
  CHECK_LE(kMaxPathLength, buf_len);

  // On OS X the executable path is saved to the stack by dyld. Reading it
  // from there is much faster than calling dladdr, especially for large
  // binaries with symbols.
  InternalScopedString exe_path(kMaxPathLength);
  uint32_t size = exe_path.size();
  if (_NSGetExecutablePath(exe_path.data(), &size) == 0 &&
      realpath(exe_path.data(), buf) != 0) {
    return internal_strlen(buf);
  }
  return 0;
}

uptr ReadLongProcessName(/*out*/char *buf, uptr buf_len) {
  return ReadBinaryName(buf, buf_len);
}

void ReExec() {
  UNIMPLEMENTED();
}

uptr GetPageSize() {
  return sysconf(_SC_PAGESIZE);
}

BlockingMutex::BlockingMutex() {
  internal_memset(this, 0, sizeof(*this));
}

void BlockingMutex::Lock() {
  CHECK(sizeof(OSSpinLock) <= sizeof(opaque_storage_));
  CHECK_EQ(OS_SPINLOCK_INIT, 0);
  CHECK_NE(owner_, (uptr)pthread_self());
  OSSpinLockLock((OSSpinLock*)&opaque_storage_);
  CHECK(!owner_);
  owner_ = (uptr)pthread_self();
}

void BlockingMutex::Unlock() {
  CHECK(owner_ == (uptr)pthread_self());
  owner_ = 0;
  OSSpinLockUnlock((OSSpinLock*)&opaque_storage_);
}

void BlockingMutex::CheckLocked() {
  CHECK_EQ((uptr)pthread_self(), owner_);
}

u64 NanoTime() {
  return 0;
}

uptr GetTlsSize() {
  return 0;
}

void InitTlsSize() {
}

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
#ifndef SANITIZER_GO
  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_addr = stack_bottom;
  *stk_size = stack_top - stack_bottom;
  *tls_addr = 0;
  *tls_size = 0;
#else
  *stk_addr = 0;
  *stk_size = 0;
  *tls_addr = 0;
  *tls_size = 0;
#endif
}

void ListOfModules::init() {
  clear();
  MemoryMappingLayout memory_mapping(false);
  memory_mapping.DumpListOfModules(&modules_);
}

bool IsHandledDeadlySignal(int signum) {
  if ((SANITIZER_WATCHOS || SANITIZER_TVOS) && !(SANITIZER_IOSSIM))
    // Handling fatal signals on watchOS and tvOS devices is disallowed.
    return false;
  return (signum == SIGSEGV || signum == SIGBUS) && common_flags()->handle_segv;
}

MacosVersion cached_macos_version = MACOS_VERSION_UNINITIALIZED;

MacosVersion GetMacosVersionInternal() {
  int mib[2] = { CTL_KERN, KERN_OSRELEASE };
  char version[100];
  uptr len = 0, maxlen = sizeof(version) / sizeof(version[0]);
  for (uptr i = 0; i < maxlen; i++) version[i] = '\0';
  // Get the version length.
  CHECK_NE(sysctl(mib, 2, 0, &len, 0, 0), -1);
  CHECK_LT(len, maxlen);
  CHECK_NE(sysctl(mib, 2, version, &len, 0, 0), -1);
  switch (version[0]) {
    case '9': return MACOS_VERSION_LEOPARD;
    case '1': {
      switch (version[1]) {
        case '0': return MACOS_VERSION_SNOW_LEOPARD;
        case '1': return MACOS_VERSION_LION;
        case '2': return MACOS_VERSION_MOUNTAIN_LION;
        case '3': return MACOS_VERSION_MAVERICKS;
        case '4': return MACOS_VERSION_YOSEMITE;
        default:
          if (IsDigit(version[1]))
            return MACOS_VERSION_UNKNOWN_NEWER;
          else
            return MACOS_VERSION_UNKNOWN;
      }
    }
    default: return MACOS_VERSION_UNKNOWN;
  }
}

MacosVersion GetMacosVersion() {
  atomic_uint32_t *cache =
      reinterpret_cast<atomic_uint32_t*>(&cached_macos_version);
  MacosVersion result =
      static_cast<MacosVersion>(atomic_load(cache, memory_order_acquire));
  if (result == MACOS_VERSION_UNINITIALIZED) {
    result = GetMacosVersionInternal();
    atomic_store(cache, result, memory_order_release);
  }
  return result;
}

uptr GetRSS() {
  struct task_basic_info info;
  unsigned count = TASK_BASIC_INFO_COUNT;
  kern_return_t result =
      task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &count);
  if (UNLIKELY(result != KERN_SUCCESS)) {
    Report("Cannot get task info. Error: %d\n", result);
    Die();
  }
  return info.resident_size;
}

void *internal_start_thread(void(*func)(void *arg), void *arg) {
  // Start the thread with signals blocked, otherwise it can steal user signals.
  __sanitizer_sigset_t set, old;
  internal_sigfillset(&set);
  internal_sigprocmask(SIG_SETMASK, &set, &old);
  pthread_t th;
  pthread_create(&th, 0, (void*(*)(void *arg))func, arg);
  internal_sigprocmask(SIG_SETMASK, &old, 0);
  return th;
}

void internal_join_thread(void *th) { pthread_join((pthread_t)th, 0); }

#ifndef SANITIZER_GO
static BlockingMutex syslog_lock(LINKER_INITIALIZED);
#endif

void WriteOneLineToSyslog(const char *s) {
#ifndef SANITIZER_GO
  syslog_lock.CheckLocked();
  asl_log(nullptr, nullptr, ASL_LEVEL_ERR, "%s", s);
#endif
}

void LogMessageOnPrintf(const char *str) {
  // Log all printf output to CrashLog.
  if (common_flags()->abort_on_error)
    CRAppendCrashLogMessage(str);
}

void LogFullErrorReport(const char *buffer) {
#ifndef SANITIZER_GO
  // Log with os_trace. This will make it into the crash log.
#if SANITIZER_OS_TRACE
  if (GetMacosVersion() >= MACOS_VERSION_YOSEMITE) {
    // os_trace requires the message (format parameter) to be a string literal.
    if (internal_strncmp(SanitizerToolName, "AddressSanitizer",
                         sizeof("AddressSanitizer") - 1) == 0)
      os_trace("Address Sanitizer reported a failure.");
    else if (internal_strncmp(SanitizerToolName, "UndefinedBehaviorSanitizer",
                              sizeof("UndefinedBehaviorSanitizer") - 1) == 0)
      os_trace("Undefined Behavior Sanitizer reported a failure.");
    else if (internal_strncmp(SanitizerToolName, "ThreadSanitizer",
                              sizeof("ThreadSanitizer") - 1) == 0)
      os_trace("Thread Sanitizer reported a failure.");
    else
      os_trace("Sanitizer tool reported a failure.");

    if (common_flags()->log_to_syslog)
      os_trace("Consult syslog for more information.");
  }
#endif

  // Log to syslog.
  // The logging on OS X may call pthread_create so we need the threading
  // environment to be fully initialized. Also, this should never be called when
  // holding the thread registry lock since that may result in a deadlock. If
  // the reporting thread holds the thread registry mutex, and asl_log waits
  // for GCD to dispatch a new thread, the process will deadlock, because the
  // pthread_create wrapper needs to acquire the lock as well.
  BlockingMutexLock l(&syslog_lock);
  if (common_flags()->log_to_syslog)
    WriteToSyslog(buffer);

  // The report is added to CrashLog as part of logging all of Printf output.
#endif
}

SignalContext::WriteFlag SignalContext::GetWriteFlag(void *context) {
#if defined(__x86_64__) || defined(__i386__)
  ucontext_t *ucontext = static_cast<ucontext_t*>(context);
  return ucontext->uc_mcontext->__es.__err & 2 /*T_PF_WRITE*/ ? WRITE : READ;
#else
  return UNKNOWN;
#endif
}

void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp) {
  ucontext_t *ucontext = (ucontext_t*)context;
# if defined(__aarch64__)
  *pc = ucontext->uc_mcontext->__ss.__pc;
#   if defined(__IPHONE_8_0) && __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_8_0
  *bp = ucontext->uc_mcontext->__ss.__fp;
#   else
  *bp = ucontext->uc_mcontext->__ss.__lr;
#   endif
  *sp = ucontext->uc_mcontext->__ss.__sp;
# elif defined(__x86_64__)
  *pc = ucontext->uc_mcontext->__ss.__rip;
  *bp = ucontext->uc_mcontext->__ss.__rbp;
  *sp = ucontext->uc_mcontext->__ss.__rsp;
# elif defined(__arm__)
  *pc = ucontext->uc_mcontext->__ss.__pc;
  *bp = ucontext->uc_mcontext->__ss.__r[7];
  *sp = ucontext->uc_mcontext->__ss.__sp;
# elif defined(__i386__)
  *pc = ucontext->uc_mcontext->__ss.__eip;
  *bp = ucontext->uc_mcontext->__ss.__ebp;
  *sp = ucontext->uc_mcontext->__ss.__esp;
# else
# error "Unknown architecture"
# endif
}

#ifndef SANITIZER_GO
static const char kDyldInsertLibraries[] = "DYLD_INSERT_LIBRARIES";
LowLevelAllocator allocator_for_env;

// Change the value of the env var |name|, leaking the original value.
// If |name_value| is NULL, the variable is deleted from the environment,
// otherwise the corresponding "NAME=value" string is replaced with
// |name_value|.
void LeakyResetEnv(const char *name, const char *name_value) {
  char **env = GetEnviron();
  uptr name_len = internal_strlen(name);
  while (*env != 0) {
    uptr len = internal_strlen(*env);
    if (len > name_len) {
      const char *p = *env;
      if (!internal_memcmp(p, name, name_len) && p[name_len] == '=') {
        // Match.
        if (name_value) {
          // Replace the old value with the new one.
          *env = const_cast<char*>(name_value);
        } else {
          // Shift the subsequent pointers back.
          char **del = env;
          do {
            del[0] = del[1];
          } while (*del++);
        }
      }
    }
    env++;
  }
}

SANITIZER_WEAK_CXX_DEFAULT_IMPL
bool ReexecDisabled() {
  return false;
}

extern "C" SANITIZER_WEAK_ATTRIBUTE double dyldVersionNumber;
static const double kMinDyldVersionWithAutoInterposition = 360.0;

bool DyldNeedsEnvVariable() {
  // Although sanitizer support was added to LLVM on OS X 10.7+, GCC users
  // still may want use them on older systems. On older Darwin platforms, dyld
  // doesn't export dyldVersionNumber symbol and we simply return true.
  if (!&dyldVersionNumber) return true;
  // If running on OS X 10.11+ or iOS 9.0+, dyld will interpose even if
  // DYLD_INSERT_LIBRARIES is not set. However, checking OS version via
  // GetMacosVersion() doesn't work for the simulator. Let's instead check
  // `dyldVersionNumber`, which is exported by dyld, against a known version
  // number from the first OS release where this appeared.
  return dyldVersionNumber < kMinDyldVersionWithAutoInterposition;
}

void MaybeReexec() {
  if (ReexecDisabled()) return;

  // Make sure the dynamic runtime library is preloaded so that the
  // wrappers work. If it is not, set DYLD_INSERT_LIBRARIES and re-exec
  // ourselves.
  Dl_info info;
  RAW_CHECK(dladdr((void*)((uptr)&__sanitizer_report_error_summary), &info));
  char *dyld_insert_libraries =
      const_cast<char*>(GetEnv(kDyldInsertLibraries));
  uptr old_env_len = dyld_insert_libraries ?
      internal_strlen(dyld_insert_libraries) : 0;
  uptr fname_len = internal_strlen(info.dli_fname);
  const char *dylib_name = StripModuleName(info.dli_fname);
  uptr dylib_name_len = internal_strlen(dylib_name);

  bool lib_is_in_env = dyld_insert_libraries &&
                       internal_strstr(dyld_insert_libraries, dylib_name);
  if (DyldNeedsEnvVariable() && !lib_is_in_env) {
    // DYLD_INSERT_LIBRARIES is not set or does not contain the runtime
    // library.
    InternalScopedString program_name(1024);
    uint32_t buf_size = program_name.size();
    _NSGetExecutablePath(program_name.data(), &buf_size);
    char *new_env = const_cast<char*>(info.dli_fname);
    if (dyld_insert_libraries) {
      // Append the runtime dylib name to the existing value of
      // DYLD_INSERT_LIBRARIES.
      new_env = (char*)allocator_for_env.Allocate(old_env_len + fname_len + 2);
      internal_strncpy(new_env, dyld_insert_libraries, old_env_len);
      new_env[old_env_len] = ':';
      // Copy fname_len and add a trailing zero.
      internal_strncpy(new_env + old_env_len + 1, info.dli_fname,
                       fname_len + 1);
      // Ok to use setenv() since the wrappers don't depend on the value of
      // asan_inited.
      setenv(kDyldInsertLibraries, new_env, /*overwrite*/1);
    } else {
      // Set DYLD_INSERT_LIBRARIES equal to the runtime dylib name.
      setenv(kDyldInsertLibraries, info.dli_fname, /*overwrite*/0);
    }
    VReport(1, "exec()-ing the program with\n");
    VReport(1, "%s=%s\n", kDyldInsertLibraries, new_env);
    VReport(1, "to enable wrappers.\n");
    execv(program_name.data(), *_NSGetArgv());

    // We get here only if execv() failed.
    Report("ERROR: The process is launched without DYLD_INSERT_LIBRARIES, "
           "which is required for the sanitizer to work. We tried to set the "
           "environment variable and re-execute itself, but execv() failed, "
           "possibly because of sandbox restrictions. Make sure to launch the "
           "executable with:\n%s=%s\n", kDyldInsertLibraries, new_env);
    RAW_CHECK("execv failed" && 0);
  }

  // Verify that interceptors really work.  We'll use dlsym to locate
  // "pthread_create", if interceptors are working, it should really point to
  // "wrap_pthread_create" within our own dylib.
  Dl_info info_pthread_create;
  void *dlopen_addr = dlsym(RTLD_DEFAULT, "pthread_create");
  RAW_CHECK(dladdr(dlopen_addr, &info_pthread_create));
  if (internal_strcmp(info.dli_fname, info_pthread_create.dli_fname) != 0) {
    Report(
        "ERROR: Interceptors are not working. This may be because %s is "
        "loaded too late (e.g. via dlopen). Please launch the executable "
        "with:\n%s=%s\n",
        SanitizerToolName, kDyldInsertLibraries, info.dli_fname);
    RAW_CHECK("interceptors not installed" && 0);
  }

  if (!lib_is_in_env)
    return;

  // DYLD_INSERT_LIBRARIES is set and contains the runtime library. Let's remove
  // the dylib from the environment variable, because interceptors are installed
  // and we don't want our children to inherit the variable.

  uptr env_name_len = internal_strlen(kDyldInsertLibraries);
  // Allocate memory to hold the previous env var name, its value, the '='
  // sign and the '\0' char.
  char *new_env = (char*)allocator_for_env.Allocate(
      old_env_len + 2 + env_name_len);
  RAW_CHECK(new_env);
  internal_memset(new_env, '\0', old_env_len + 2 + env_name_len);
  internal_strncpy(new_env, kDyldInsertLibraries, env_name_len);
  new_env[env_name_len] = '=';
  char *new_env_pos = new_env + env_name_len + 1;

  // Iterate over colon-separated pieces of |dyld_insert_libraries|.
  char *piece_start = dyld_insert_libraries;
  char *piece_end = NULL;
  char *old_env_end = dyld_insert_libraries + old_env_len;
  do {
    if (piece_start[0] == ':') piece_start++;
    piece_end = internal_strchr(piece_start, ':');
    if (!piece_end) piece_end = dyld_insert_libraries + old_env_len;
    if ((uptr)(piece_start - dyld_insert_libraries) > old_env_len) break;
    uptr piece_len = piece_end - piece_start;

    char *filename_start =
        (char *)internal_memrchr(piece_start, '/', piece_len);
    uptr filename_len = piece_len;
    if (filename_start) {
      filename_start += 1;
      filename_len = piece_len - (filename_start - piece_start);
    } else {
      filename_start = piece_start;
    }

    // If the current piece isn't the runtime library name,
    // append it to new_env.
    if ((dylib_name_len != filename_len) ||
        (internal_memcmp(filename_start, dylib_name, dylib_name_len) != 0)) {
      if (new_env_pos != new_env + env_name_len + 1) {
        new_env_pos[0] = ':';
        new_env_pos++;
      }
      internal_strncpy(new_env_pos, piece_start, piece_len);
      new_env_pos += piece_len;
    }
    // Move on to the next piece.
    piece_start = piece_end;
  } while (piece_start < old_env_end);

  // Can't use setenv() here, because it requires the allocator to be
  // initialized.
  // FIXME: instead of filtering DYLD_INSERT_LIBRARIES here, do it in
  // a separate function called after InitializeAllocator().
  if (new_env_pos == new_env + env_name_len + 1) new_env = NULL;
  LeakyResetEnv(kDyldInsertLibraries, new_env);
}
#endif  // SANITIZER_GO

char **GetArgv() {
  return *_NSGetArgv();
}

// FIXME implement on this platform.
void GetMemoryProfile(fill_profile_f cb, uptr *stats, uptr stats_size) { }

}  // namespace __sanitizer

#endif  // SANITIZER_MAC
