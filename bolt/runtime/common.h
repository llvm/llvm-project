//===- bolt/runtime/common.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if !defined(__APPLE__)

#include <cstddef>
#include <cstdint>

#include "config.h"

#ifdef HAVE_ELF_H
#include <elf.h>
#endif

#else

typedef __SIZE_TYPE__ size_t;
#define __SSIZE_TYPE__                                                         \
  __typeof__(_Generic((__SIZE_TYPE__)0, unsigned long long int                 \
                      : (long long int)0, unsigned long int                    \
                      : (long int)0, unsigned int                              \
                      : (int)0, unsigned short                                 \
                      : (short)0, unsigned char                                \
                      : (signed char)0))
typedef __SSIZE_TYPE__ ssize_t;

typedef unsigned long long uint64_t;
typedef unsigned uint32_t;
typedef unsigned char uint8_t;

typedef long long int64_t;
typedef int int32_t;

#endif

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                                                               \
  "push %%rax\n"                                                               \
  "push %%rbx\n"                                                               \
  "push %%rcx\n"                                                               \
  "push %%rdx\n"                                                               \
  "push %%rdi\n"                                                               \
  "push %%rsi\n"                                                               \
  "push %%rbp\n"                                                               \
  "push %%r8\n"                                                                \
  "push %%r9\n"                                                                \
  "push %%r10\n"                                                               \
  "push %%r11\n"                                                               \
  "push %%r12\n"                                                               \
  "push %%r13\n"                                                               \
  "push %%r14\n"                                                               \
  "push %%r15\n"                                                               \
  "sub $8, %%rsp\n"

// Mirrors SAVE_ALL
#define RESTORE_ALL                                                            \
  "add $8, %%rsp\n"                                                            \
  "pop %%r15\n"                                                                \
  "pop %%r14\n"                                                                \
  "pop %%r13\n"                                                                \
  "pop %%r12\n"                                                                \
  "pop %%r11\n"                                                                \
  "pop %%r10\n"                                                                \
  "pop %%r9\n"                                                                 \
  "pop %%r8\n"                                                                 \
  "pop %%rbp\n"                                                                \
  "pop %%rsi\n"                                                                \
  "pop %%rdi\n"                                                                \
  "pop %%rdx\n"                                                                \
  "pop %%rcx\n"                                                                \
  "pop %%rbx\n"                                                                \
  "pop %%rax\n"

// Anonymous namespace covering everything but our library entry point
namespace {

constexpr uint32_t BufSize = 10240;

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

uint64_t __read(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
#if defined(__APPLE__)
#define READ_SYSCALL 0x2000003
#else
#define READ_SYSCALL 0
#endif
  __asm__ __volatile__("movq $" STRINGIFY(READ_SYSCALL) ", %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd), "S"(buf), "d"(count)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __write(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
#if defined(__APPLE__)
#define WRITE_SYSCALL 0x2000004
#else
#define WRITE_SYSCALL 1
#endif
  __asm__ __volatile__("movq $" STRINGIFY(WRITE_SYSCALL) ", %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd), "S"(buf), "d"(count)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

void *__mmap(uint64_t addr, uint64_t size, uint64_t prot, uint64_t flags,
             uint64_t fd, uint64_t offset) {
#if defined(__APPLE__)
#define MMAP_SYSCALL 0x20000c5
#else
#define MMAP_SYSCALL 9
#endif
  void *ret;
  register uint64_t r8 asm("r8") = fd;
  register uint64_t r9 asm("r9") = offset;
  register uint64_t r10 asm("r10") = flags;
  __asm__ __volatile__("movq $" STRINGIFY(MMAP_SYSCALL) ", %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(size), "d"(prot), "r"(r10), "r"(r8),
                         "r"(r9)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __munmap(void *addr, uint64_t size) {
#if defined(__APPLE__)
#define MUNMAP_SYSCALL 0x2000049
#else
#define MUNMAP_SYSCALL 11
#endif
  uint64_t ret;
  __asm__ __volatile__("movq $" STRINGIFY(MUNMAP_SYSCALL) ", %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(size)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

#define SIG_BLOCK 0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2

static const uint64_t MaskAllSignals[] = {-1ULL};

uint64_t __sigprocmask(int how, const void *set, void *oldset) {
#if defined(__APPLE__)
#define SIGPROCMASK_SYSCALL 0x2000030
#else
#define SIGPROCMASK_SYSCALL 14
#endif
  uint64_t ret;
  register long r10 asm("r10") = sizeof(uint64_t);
  __asm__ __volatile__("movq $" STRINGIFY(SIGPROCMASK_SYSCALL) ", %%rax\n"
                                                               "syscall\n"
                       : "=a"(ret)
                       : "D"(how), "S"(set), "d"(oldset), "r"(r10)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __exit(uint64_t code) {
#if defined(__APPLE__)
#define EXIT_SYSCALL 0x2000001
#else
#define EXIT_SYSCALL 231
#endif
  uint64_t ret;
  __asm__ __volatile__("movq $" STRINGIFY(EXIT_SYSCALL) ", %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(code)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

// Helper functions for writing strings to the .fdata file. We intentionally
// avoid using libc names (lowercase memset) to make it clear it is our impl.

/// Write number Num using Base to the buffer in OutBuf, returns a pointer to
/// the end of the string.
char *intToStr(char *OutBuf, uint64_t Num, uint32_t Base) {
  const char *Chars = "0123456789abcdef";
  char Buf[21];
  char *Ptr = Buf;
  while (Num) {
    *Ptr++ = *(Chars + (Num % Base));
    Num /= Base;
  }
  if (Ptr == Buf) {
    *OutBuf++ = '0';
    return OutBuf;
  }
  while (Ptr != Buf)
    *OutBuf++ = *--Ptr;

  return OutBuf;
}

/// Copy Str to OutBuf, returns a pointer to the end of the copied string
char *strCopy(char *OutBuf, const char *Str, int32_t Size = BufSize) {
  while (*Str) {
    *OutBuf++ = *Str++;
    if (--Size <= 0)
      return OutBuf;
  }
  return OutBuf;
}

/// Compare two strings, at most Num bytes.
int strnCmp(const char *Str1, const char *Str2, size_t Num) {
  while (Num && *Str1 && (*Str1 == *Str2)) {
    Num--;
    Str1++;
    Str2++;
  }
  if (Num == 0)
    return 0;
  return *(unsigned char *)Str1 - *(unsigned char *)Str2;
}

void memSet(char *Buf, char C, uint32_t Size) {
  for (int I = 0; I < Size; ++I)
    *Buf++ = C;
}

void *memCpy(void *Dest, const void *Src, size_t Len) {
  char *d = static_cast<char *>(Dest);
  const char *s = static_cast<const char *>(Src);
  while (Len--)
    *d++ = *s++;
  return Dest;
}

uint32_t strLen(const char *Str) {
  uint32_t Size = 0;
  while (*Str++)
    ++Size;
  return Size;
}

void reportNumber(const char *Msg, uint64_t Num, uint32_t Base) {
  char Buf[BufSize];
  char *Ptr = Buf;
  Ptr = strCopy(Ptr, Msg, BufSize - 23);
  Ptr = intToStr(Ptr, Num, Base);
  Ptr = strCopy(Ptr, "\n");
  __write(2, Buf, Ptr - Buf);
}

void report(const char *Msg) { __write(2, Msg, strLen(Msg)); }

unsigned long hexToLong(const char *Str, char Terminator = '\0') {
  unsigned long Res = 0;
  while (*Str != Terminator) {
    Res <<= 4;
    if ('0' <= *Str && *Str <= '9')
      Res += *Str++ - '0';
    else if ('a' <= *Str && *Str <= 'f')
      Res += *Str++ - 'a' + 10;
    else if ('A' <= *Str && *Str <= 'F')
      Res += *Str++ - 'A' + 10;
    else
      return 0;
  }
  return Res;
}

#if !defined(__APPLE__)
// We use a stack-allocated buffer for string manipulation in many pieces of
// this code, including the code that prints each line of the fdata file. This
// buffer needs to accomodate large function names, but shouldn't be arbitrarily
// large (dynamically allocated) for simplicity of our memory space usage.

// Declare some syscall wrappers we use throughout this code to avoid linking
// against system libc.
uint64_t __open(const char *pathname, uint64_t flags, uint64_t mode) {
  uint64_t ret;
  __asm__ __volatile__("movq $2, %%rax\n"
                       "syscall"
                       : "=a"(ret)
                       : "D"(pathname), "S"(flags), "d"(mode)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

struct dirent {
  unsigned long d_ino;     /* Inode number */
  unsigned long d_off;     /* Offset to next linux_dirent */
  unsigned short d_reclen; /* Length of this linux_dirent */
  char d_name[];           /* Filename (null-terminated) */
                           /* length is actually (d_reclen - 2 -
                             offsetof(struct linux_dirent, d_name)) */
};

long __getdents(unsigned int fd, dirent *dirp, size_t count) {
  long ret;
  __asm__ __volatile__("movq $78, %%rax\n"
                       "syscall"
                       : "=a"(ret)
                       : "D"(fd), "S"(dirp), "d"(count)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __readlink(const char *pathname, char *buf, size_t bufsize) {
  uint64_t ret;
  __asm__ __volatile__("movq $89, %%rax\n"
                       "syscall"
                       : "=a"(ret)
                       : "D"(pathname), "S"(buf), "d"(bufsize)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __lseek(uint64_t fd, uint64_t pos, uint64_t whence) {
  uint64_t ret;
  __asm__ __volatile__("movq $8, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd), "S"(pos), "d"(whence)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __close(uint64_t fd) {
  uint64_t ret;
  __asm__ __volatile__("movq $3, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __madvise(void *addr, size_t length, int advice) {
  int ret;
  __asm__ __volatile__("movq $28, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(length), "d"(advice)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

struct timespec {
  uint64_t tv_sec;  /* seconds */
  uint64_t tv_nsec; /* nanoseconds */
};

uint64_t __nanosleep(const timespec *req, timespec *rem) {
  uint64_t ret;
  __asm__ __volatile__("movq $35, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(req), "S"(rem)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int64_t __fork() {
  uint64_t ret;
  __asm__ __volatile__("movq $57, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       :
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __mprotect(void *addr, size_t len, int prot) {
  int ret;
  __asm__ __volatile__("movq $10, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(len), "d"(prot)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __getpid() {
  uint64_t ret;
  __asm__ __volatile__("movq $39, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       :
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __getppid() {
  uint64_t ret;
  __asm__ __volatile__("movq $110, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       :
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __setpgid(uint64_t pid, uint64_t pgid) {
  int ret;
  __asm__ __volatile__("movq $109, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(pid), "S"(pgid)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __getpgid(uint64_t pid) {
  uint64_t ret;
  __asm__ __volatile__("movq $121, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(pid)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __kill(uint64_t pid, int sig) {
  int ret;
  __asm__ __volatile__("movq $62, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(pid), "S"(sig)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __fsync(int fd) {
  int ret;
  __asm__ __volatile__("movq $74, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

#endif

void reportError(const char *Msg, uint64_t Size) {
  __write(2, Msg, Size);
  __exit(1);
}

void assert(bool Assertion, const char *Msg) {
  if (Assertion)
    return;
  char Buf[BufSize];
  char *Ptr = Buf;
  Ptr = strCopy(Ptr, "Assertion failed: ");
  Ptr = strCopy(Ptr, Msg, BufSize - 40);
  Ptr = strCopy(Ptr, "\n");
  reportError(Buf, Ptr - Buf);
}

class Mutex {
  volatile bool InUse{false};

public:
  bool acquire() { return !__atomic_test_and_set(&InUse, __ATOMIC_ACQUIRE); }
  void release() { __atomic_clear(&InUse, __ATOMIC_RELEASE); }
};

/// RAII wrapper for Mutex
class Lock {
  Mutex &M;
  uint64_t SignalMask[1] = {};

public:
  Lock(Mutex &M) : M(M) {
    __sigprocmask(SIG_BLOCK, MaskAllSignals, SignalMask);
    while (!M.acquire()) {
    }
  }

  ~Lock() {
    M.release();
    __sigprocmask(SIG_SETMASK, SignalMask, nullptr);
  }
};

/// RAII wrapper for Mutex
class TryLock {
  Mutex &M;
  bool Locked = false;

public:
  TryLock(Mutex &M) : M(M) {
    int Retry = 100;
    while (--Retry && !M.acquire())
      ;
    if (Retry)
      Locked = true;
  }
  bool isLocked() { return Locked; }

  ~TryLock() {
    if (isLocked())
      M.release();
  }
};

inline uint64_t alignTo(uint64_t Value, uint64_t Align) {
  return (Value + Align - 1) / Align * Align;
}

} // anonymous namespace
