//===-- scudo_utils.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Platform specific utility functions.
///
//===----------------------------------------------------------------------===//

#include "scudo_utils.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <unistd.h>

#include <cstring>

// TODO(kostyak): remove __sanitizer *Printf uses in favor for our own less
//                complicated string formatting code. The following is a
//                temporary workaround to be able to use __sanitizer::VSNPrintf.
namespace __sanitizer {

extern int VSNPrintf(char *buff, int buff_length, const char *format,
                     va_list args);

} // namespace __sanitizer

namespace __scudo {

FORMAT(1, 2)
void dieWithMessage(const char *Format, ...) {
  // Our messages are tiny, 128 characters is more than enough.
  char Message[128];
  va_list Args;
  va_start(Args, Format);
  __sanitizer::VSNPrintf(Message, sizeof(Message), Format, Args);
  va_end(Args);
  RawWrite(Message);
  Die();
}

typedef struct {
  u32 Eax;
  u32 Ebx;
  u32 Ecx;
  u32 Edx;
} CPUIDInfo;

static void getCPUID(CPUIDInfo *info, u32 leaf, u32 subleaf)
{
  asm volatile("cpuid"
      : "=a" (info->Eax), "=b" (info->Ebx), "=c" (info->Ecx), "=d" (info->Edx)
      : "a" (leaf), "c" (subleaf)
  );
}

// Returns true is the CPU is a "GenuineIntel" or "AuthenticAMD"
static bool isSupportedCPU()
{
  CPUIDInfo Info;

  getCPUID(&Info, 0, 0);
  if (memcmp(reinterpret_cast<char *>(&Info.Ebx), "Genu", 4) == 0 &&
      memcmp(reinterpret_cast<char *>(&Info.Edx), "ineI", 4) == 0 &&
      memcmp(reinterpret_cast<char *>(&Info.Ecx), "ntel", 4) == 0) {
      return true;
  }
  if (memcmp(reinterpret_cast<char *>(&Info.Ebx), "Auth", 4) == 0 &&
      memcmp(reinterpret_cast<char *>(&Info.Edx), "enti", 4) == 0 &&
      memcmp(reinterpret_cast<char *>(&Info.Ecx), "cAMD", 4) == 0) {
      return true;
  }
  return false;
}

bool testCPUFeature(CPUFeature feature)
{
  static bool InfoInitialized = false;
  static CPUIDInfo CPUInfo = {};

  if (InfoInitialized == false) {
    if (isSupportedCPU() == true)
      getCPUID(&CPUInfo, 1, 0);
    else
      UNIMPLEMENTED();
    InfoInitialized = true;
  }
  switch (feature) {
    case SSE4_2:
      return ((CPUInfo.Ecx >> 20) & 0x1) != 0;
    default:
      break;
  }
  return false;
}

// readRetry will attempt to read Count bytes from the Fd specified, and if
// interrupted will retry to read additional bytes to reach Count.
static ssize_t readRetry(int Fd, u8 *Buffer, size_t Count) {
  ssize_t AmountRead = 0;
  while (static_cast<size_t>(AmountRead) < Count) {
    ssize_t Result = read(Fd, Buffer + AmountRead, Count - AmountRead);
    if (Result > 0)
      AmountRead += Result;
    else if (!Result)
      break;
    else if (errno != EINTR) {
      AmountRead = -1;
      break;
    }
  }
  return AmountRead;
}

// Default constructor for Xorshift128Plus seeds the state with /dev/urandom
Xorshift128Plus::Xorshift128Plus() {
  int Fd = open("/dev/urandom", O_RDONLY);
  bool Success = readRetry(Fd, reinterpret_cast<u8 *>(&State_0_),
                           sizeof(State_0_)) == sizeof(State_0_);
  Success &= readRetry(Fd, reinterpret_cast<u8 *>(&State_1_),
                           sizeof(State_1_)) == sizeof(State_1_);
  close(Fd);
  if (!Success) {
    dieWithMessage("ERROR: failed to read enough data from /dev/urandom.\n");
  }
}

} // namespace __scudo
