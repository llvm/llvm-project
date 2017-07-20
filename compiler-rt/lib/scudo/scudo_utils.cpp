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
#if defined(__x86_64__) || defined(__i386__)
# include <cpuid.h>
#endif
#if defined(__arm__) || defined(__aarch64__)
# include <sys/auxv.h>
#endif

// TODO(kostyak): remove __sanitizer *Printf uses in favor for our own less
//                complicated string formatting code. The following is a
//                temporary workaround to be able to use __sanitizer::VSNPrintf.
namespace __sanitizer {

extern int VSNPrintf(char *buff, int buff_length, const char *format,
                     va_list args);

}  // namespace __sanitizer

namespace __scudo {

FORMAT(1, 2)
void NORETURN dieWithMessage(const char *Format, ...) {
  // Our messages are tiny, 256 characters is more than enough.
  char Message[256];
  va_list Args;
  va_start(Args, Format);
  __sanitizer::VSNPrintf(Message, sizeof(Message), Format, Args);
  va_end(Args);
  RawWrite(Message);
  Die();
}

#if defined(__x86_64__) || defined(__i386__)
// i386 and x86_64 specific code to detect CRC32 hardware support via CPUID.
// CRC32 requires the SSE 4.2 instruction set.
typedef struct {
  u32 Eax;
  u32 Ebx;
  u32 Ecx;
  u32 Edx;
} CPUIDRegs;

static void getCPUID(CPUIDRegs *Regs, u32 Level)
{
  __get_cpuid(Level, &Regs->Eax, &Regs->Ebx, &Regs->Ecx, &Regs->Edx);
}

CPUIDRegs getCPUFeatures() {
  CPUIDRegs VendorRegs = {};
  getCPUID(&VendorRegs, 0);
  bool IsIntel =
      (VendorRegs.Ebx == signature_INTEL_ebx) &&
      (VendorRegs.Edx == signature_INTEL_edx) &&
      (VendorRegs.Ecx == signature_INTEL_ecx);
  bool IsAMD =
      (VendorRegs.Ebx == signature_AMD_ebx) &&
      (VendorRegs.Edx == signature_AMD_edx) &&
      (VendorRegs.Ecx == signature_AMD_ecx);
  // Default to an empty feature set if not on a supported CPU.
  CPUIDRegs FeaturesRegs = {};
  if (IsIntel || IsAMD) {
    getCPUID(&FeaturesRegs, 1);
  }
  return FeaturesRegs;
}

#ifndef bit_SSE4_2
# define bit_SSE4_2 bit_SSE42  // clang and gcc have different defines.
#endif

bool testCPUFeature(CPUFeature Feature)
{
  CPUIDRegs FeaturesRegs = getCPUFeatures();

  switch (Feature) {
    case CRC32CPUFeature:  // CRC32 is provided by SSE 4.2.
      return !!(FeaturesRegs.Ecx & bit_SSE4_2);
    default:
      break;
  }
  return false;
}
#elif defined(__arm__) || defined(__aarch64__)
// For ARM and AArch64, hardware CRC32 support is indicated in the
// AT_HWVAL auxiliary vector.

#ifndef HWCAP_CRC32
# define HWCAP_CRC32 (1<<7)  // HWCAP_CRC32 is missing on older platforms.
#endif

bool testCPUFeature(CPUFeature Feature) {
  uptr HWCap = getauxval(AT_HWCAP);

  switch (Feature) {
    case CRC32CPUFeature:
      return !!(HWCap & HWCAP_CRC32);
    default:
      break;
  }
  return false;
}
#else
bool testCPUFeature(CPUFeature Feature) {
  return false;
}
#endif  // defined(__x86_64__) || defined(__i386__)

}  // namespace __scudo
