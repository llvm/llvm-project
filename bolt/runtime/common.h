//===- bolt/runtime/common.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "runtime_types.h"
#include "syscall_wrappers.h"

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

// We use a stack-allocated buffer for string manipulation in many pieces of
// this code, including the code that prints each line of the fdata file. This
// buffer needs to accommodate large function names, but shouldn't be
// arbitrarily large (dynamically allocated) for simplicity of our memory space
// usage.
constexpr uint32_t BufSize = 32768U;

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
#if !defined(__ANDROID__)
  char Buf[BufSize];
  char *Ptr = Buf;
  Ptr = strCopy(Ptr, Msg, BufSize - 23);
  Ptr = intToStr(Ptr, Num, Base);
  Ptr = strCopy(Ptr, "\n");
  __write(2, Buf, Ptr - Buf);
#endif
}

void report(const char *Msg) {
#if !defined(__ANDROID__)
  __write(2, Msg, strLen(Msg));
#endif
}

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
#if !defined(__ANDROID__)
  __write(2, Msg, Size);
#endif
  __exit(1);
}

void assert(bool Assertion, const char *Msg) {
  if (Assertion)
    return;
#if defined(__ANDROID__)
  __exit(1);
#else
  char Buf[BufSize];
  char *Ptr = Buf;
  Ptr = strCopy(Ptr, "Assertion failed: ");
  Ptr = strCopy(Ptr, Msg, BufSize - 40);
  Ptr = strCopy(Ptr, "\n");
  reportError(Buf, Ptr - Buf);
#endif
}

static const sigset_t MaskAllSignals[] = {-1ULL};

class Mutex {
  volatile bool InUse{false};

public:
  bool acquire() { return !__atomic_test_and_set(&InUse, __ATOMIC_ACQUIRE); }
  void release() { __atomic_clear(&InUse, __ATOMIC_RELEASE); }
};

/// RAII wrapper for Mutex
class Lock {
  Mutex &M;
  sigset_t SignalMask[1] = {};

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

constexpr intptr_t MaxErrno = 4095;
// The function is used to detect errors from syscall wrappers that return
// pointers instead of scalar values (for example, __mmap).
// The return value of the __mmap wrapper is either a valid address or an error
// (a negative value), not MAP_FAILED macro. The MAP_FAILED macro is a libc
// return value of the mmap library function in case of an error where the
// actual error is returned via errno variable. So, it is incorrect to compare
// the __mmap syscall wrapper return value with MAP_FAILED, as only the EPERM
// (-1) error is checked.
inline bool isErrValue(const void *Val) {
  const intptr_t PtrVal = reinterpret_cast<intptr_t>(Val);
  return PtrVal >= -MaxErrno && PtrVal <= -1;
}

} // anonymous namespace
