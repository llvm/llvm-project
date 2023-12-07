//===- bolt/runtime/common.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__linux__)

#include <cstddef>
#include <cstdint>

#include "config.h"

#ifdef HAVE_ELF_H
#include <elf.h>
#endif

#elif defined(__APPLE__)

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

#else
#error "For Linux or MacOS only"
#endif

#define PROT_READ 0x1  /* Page can be read.  */
#define PROT_WRITE 0x2 /* Page can be written.  */
#define PROT_EXEC 0x4  /* Page can be executed.  */
#define PROT_NONE 0x0  /* Page can not be accessed.  */
#define PROT_GROWSDOWN                                                         \
  0x01000000 /* Extend change to start of                                      \
                growsdown vma (mprotect only).  */
#define PROT_GROWSUP                                                           \
  0x02000000 /* Extend change to start of                                      \
                growsup vma (mprotect only).  */

/* Sharing types (must choose one and only one of these).  */
#define MAP_SHARED 0x01  /* Share changes.  */
#define MAP_PRIVATE 0x02 /* Changes are private.  */
#define MAP_FIXED 0x10   /* Interpret addr exactly.  */

#if defined(__APPLE__)
#define MAP_ANONYMOUS 0x1000
#else
#define MAP_ANONYMOUS 0x20
#endif

#define MAP_FAILED ((void *)-1)

#define SEEK_SET 0 /* Seek from beginning of file.  */
#define SEEK_CUR 1 /* Seek from current position.  */
#define SEEK_END 2 /* Seek from end of file.  */

#define O_RDONLY 0
#define O_WRONLY 1
#define O_RDWR 2
#define O_CREAT 64
#define O_TRUNC 512
#define O_APPEND 1024

// Functions that are required by freestanding environment. Compiler may
// generate calls to these implicitly.
extern "C" {
void *memcpy(void *Dest, const void *Src, size_t Len) {
  uint8_t *d = static_cast<uint8_t *>(Dest);
  const uint8_t *s = static_cast<const uint8_t *>(Src);
  while (Len--)
    *d++ = *s++;
  return Dest;
}

void *memmove(void *Dest, const void *Src, size_t Len) {
  uint8_t *d = static_cast<uint8_t *>(Dest);
  const uint8_t *s = static_cast<const uint8_t *>(Src);
  if (d < s) {
    while (Len--)
      *d++ = *s++;
  } else {
    s += Len - 1;
    d += Len - 1;
    while (Len--)
      *d-- = *s--;
  }

  return Dest;
}

void *memset(void *Buf, int C, size_t Size) {
  char *S = (char *)Buf;
  for (size_t I = 0; I < Size; ++I)
    *S++ = C;
  return Buf;
}

int memcmp(const void *s1, const void *s2, size_t n) {
  const uint8_t *c1 = static_cast<const uint8_t *>(s1);
  const uint8_t *c2 = static_cast<const uint8_t *>(s2);
  for (; n--; c1++, c2++) {
    if (*c1 != *c2)
      return *c1 < *c2 ? -1 : 1;
  }
  return 0;
}
} // extern "C"

// Anonymous namespace covering everything but our library entry point
namespace {

struct dirent64 {
  uint64_t d_ino;          /* Inode number */
  int64_t d_off;           /* Offset to next linux_dirent */
  unsigned short d_reclen; /* Length of this linux_dirent */
  unsigned char d_type;
  char d_name[]; /* Filename (null-terminated) */
                 /* length is actually (d_reclen - 2 -
                   offsetof(struct linux_dirent, d_name)) */
};

/* Length of the entries in `struct utsname' is 65.  */
#define _UTSNAME_LENGTH 65

struct UtsNameTy {
  char sysname[_UTSNAME_LENGTH];  /* Operating system name (e.g., "Linux") */
  char nodename[_UTSNAME_LENGTH]; /* Name within "some implementation-defined
                      network" */
  char release[_UTSNAME_LENGTH]; /* Operating system release (e.g., "2.6.28") */
  char version[_UTSNAME_LENGTH]; /* Operating system version */
  char machine[_UTSNAME_LENGTH]; /* Hardware identifier */
  char domainname[_UTSNAME_LENGTH]; /* NIS or YP domain name */
};

struct timespec {
  uint64_t tv_sec;  /* seconds */
  uint64_t tv_nsec; /* nanoseconds */
};

#if defined(__aarch64__)
#include "sys_aarch64.h"
#else
#include "sys_x86_64.h"
#endif

constexpr uint32_t BufSize = 10240;

// Helper functions for writing strings to the .fdata file. We intentionally
// avoid using libc names to make it clear it is our impl.

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

uint32_t strLen(const char *Str) {
  uint32_t Size = 0;
  while (*Str++)
    ++Size;
  return Size;
}

void *strStr(const char *const Haystack, const char *const Needle) {
  int j = 0;

  for (int i = 0; i < strLen(Haystack); i++) {
    if (Haystack[i] == Needle[0]) {
      for (j = 1; j < strLen(Needle); j++) {
        if (Haystack[i + j] != Needle[j])
          break;
      }
      if (j == strLen(Needle))
        return (void *)&Haystack[i];
    }
  }
  return nullptr;
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

/// Starting from character at \p buf, find the longest consecutive sequence
/// of digits (0-9) and convert it to uint32_t. The converted value
/// is put into \p ret. \p end marks the end of the buffer to avoid buffer
/// overflow. The function \returns whether a valid uint32_t value is found.
/// \p buf will be updated to the next character right after the digits.
static bool scanUInt32(const char *&Buf, const char *End, uint32_t &Ret) {
  uint64_t Result = 0;
  const char *OldBuf = Buf;
  while (Buf < End && ((*Buf) >= '0' && (*Buf) <= '9')) {
    Result = Result * 10 + (*Buf) - '0';
    ++Buf;
  }
  if (OldBuf != Buf && Result <= 0xFFFFFFFFu) {
    Ret = static_cast<uint32_t>(Result);
    return true;
  }
  return false;
}

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

#define SIG_BLOCK 0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2

static const uint64_t MaskAllSignals[] = {-1ULL};

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
