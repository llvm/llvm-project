//===-- windows/PosixApi.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_windows_PosixApi_h
#define liblldb_Host_windows_PosixApi_h

#include "llvm/Support/Compiler.h"
#if !defined(LLVM_ON_WIN32)
#error "windows/PosixApi.h being #included on non Windows system!"
#endif

// va_start, va_end, etc macros.
#include <stdarg.h>

// time_t, timespec, etc.
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 32768
#endif

#define O_NOCTTY 0
#define O_NONBLOCK 0
#define SIGTRAP 5
#define SIGKILL 9
#define SIGSTOP 20

#if defined(_MSC_VER)
#define S_IRUSR S_IREAD  /* read, user */
#define S_IWUSR S_IWRITE /* write, user */
#define S_IXUSR 0        /* execute, user */
#endif
#define S_IRGRP 0 /* read, group */
#define S_IWGRP 0 /* write, group */
#define S_IXGRP 0 /* execute, group */
#define S_IROTH 0 /* read, others */
#define S_IWOTH 0 /* write, others */
#define S_IXOTH 0 /* execute, others */
#define S_IRWXU 0
#define S_IRWXG 0
#define S_IRWXO 0

#ifdef _MSC_VER

// PRIxxx format macros for printf()
#include <inttypes.h>

// open(), close(), creat(), etc.
#include <io.h>

typedef unsigned short mode_t;

// pyconfig.h typedefs this.  We require python headers to be included before
// any
// LLDB headers, but there's no way to prevent python's pid_t definition from
// leaking, so this is the best option.
#ifndef Py_CONFIG_H
typedef uint32_t pid_t;
#endif

#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2

#define S_IFDIR _S_IFDIR
#define S_ISDIR(mode) (((mode)&S_IFMT) == S_IFDIR)

#endif // _MSC_VER

// MSVC 2015 and higher have timespec.  Otherwise we need to define it
// ourselves.
#if !defined(_MSC_VER) || _MSC_VER < 1900
struct timespec {
  time_t tv_sec;
  long tv_nsec;
};
#endif

// Various useful posix functions that are not present in Windows.  We provide
// custom implementations.
int vasprintf(char **ret, const char *fmt, va_list ap);
char *strcasestr(const char *s, const char *find);
char *realpath(const char *name, char *resolved);

int usleep(uint32_t useconds);
char *getcwd(char *path, int max);
int chdir(const char *path);
char *basename(char *path);
char *dirname(char *path);

int strcasecmp(const char *s1, const char *s2);
int strncasecmp(const char *s1, const char *s2, size_t n);

// empty functions
inline int posix_openpt(int flag) { LLVM_BUILTIN_UNREACHABLE; }

inline int strerror_r(int errnum, char *buf, size_t buflen) {
  LLVM_BUILTIN_UNREACHABLE;
}

inline int unlockpt(int fd) { LLVM_BUILTIN_UNREACHABLE; }
inline int grantpt(int fd) { LLVM_BUILTIN_UNREACHABLE; }
inline char *ptsname(int fd) { LLVM_BUILTIN_UNREACHABLE; }

inline pid_t fork(void) { LLVM_BUILTIN_UNREACHABLE; }
inline pid_t setsid(void) { LLVM_BUILTIN_UNREACHABLE; }

// vsnprintf and snprintf are provided in MSVC 2015 and higher.
#if _MSC_VER < 1900
namespace lldb_private {
int vsnprintf(char *buffer, size_t count, const char *format, va_list argptr);
}

// inline to avoid linkage conflicts
int inline snprintf(char *buffer, size_t count, const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  int r = lldb_private::vsnprintf(buffer, count, format, argptr);
  va_end(argptr);
  return r;
}
#endif

#endif
