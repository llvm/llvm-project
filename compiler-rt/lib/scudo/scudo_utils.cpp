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

#if defined(__x86_64__) || defined(__i386__)
# include <cpuid.h>
#elif defined(__arm__) || defined(__aarch64__)
# include "sanitizer_common/sanitizer_getauxval.h"
# if SANITIZER_POSIX
#  include "sanitizer_common/sanitizer_posix.h"
#  include <fcntl.h>
# endif
#endif

#include <stdarg.h>

// TODO(kostyak): remove __sanitizer *Printf uses in favor for our own less
//                complicated string formatting code. The following is a
//                temporary workaround to be able to use __sanitizer::VSNPrintf.
namespace __sanitizer {

extern int VSNPrintf(char *buff, int buff_length, const char *format,
                     va_list args);

}  // namespace __sanitizer

namespace __scudo {

FORMAT(1, 2) void NORETURN dieWithMessage(const char *Format, ...) {
  // Our messages are tiny, 256 characters is more than enough.
  char Message[256];
  va_list Args;
  va_start(Args, Format);
  VSNPrintf(Message, sizeof(Message), Format, Args);
  va_end(Args);
  RawWrite(Message);
  Die();
}

#if defined(__x86_64__) || defined(__i386__)
// i386 and x86_64 specific code to detect CRC32 hardware support via CPUID.
// CRC32 requires the SSE 4.2 instruction set.
# ifndef bit_SSE4_2
#  define bit_SSE4_2 bit_SSE42  // clang and gcc have different defines.
# endif
bool hasHardwareCRC32() {
  u32 Eax, Ebx, Ecx, Edx;
  __get_cpuid(0, &Eax, &Ebx, &Ecx, &Edx);
  const bool IsIntel = (Ebx == signature_INTEL_ebx) &&
                       (Edx == signature_INTEL_edx) &&
                       (Ecx == signature_INTEL_ecx);
  const bool IsAMD = (Ebx == signature_AMD_ebx) &&
                     (Edx == signature_AMD_edx) &&
                     (Ecx == signature_AMD_ecx);
  if (!IsIntel && !IsAMD)
    return false;
  __get_cpuid(1, &Eax, &Ebx, &Ecx, &Edx);
  return !!(Ecx & bit_SSE4_2);
}
#elif defined(__arm__) || defined(__aarch64__)
// For ARM and AArch64, hardware CRC32 support is indicated in the AT_HWCAP
// auxiliary vector.
# ifndef AT_HWCAP
#  define AT_HWCAP 16
# endif
# ifndef HWCAP_CRC32
#  define HWCAP_CRC32 (1 << 7)  // HWCAP_CRC32 is missing on older platforms.
# endif
# if SANITIZER_POSIX
bool hasHardwareCRC32ARMPosix() {
  uptr F = internal_open("/proc/self/auxv", O_RDONLY);
  if (internal_iserror(F))
    return false;
  struct { uptr Tag; uptr Value; } Entry = { 0, 0 };
  for (;;) {
    uptr N = internal_read(F, &Entry, sizeof(Entry));
    if (internal_iserror(N) || N != sizeof(Entry) ||
        (Entry.Tag == 0 && Entry.Value == 0) || Entry.Tag == AT_HWCAP)
      break;
  }
  internal_close(F);
  return (Entry.Tag == AT_HWCAP && (Entry.Value & HWCAP_CRC32) != 0);
}
# else
bool hasHardwareCRC32ARMPosix() { return false; }
# endif  // SANITIZER_POSIX

bool hasHardwareCRC32() {
  if (&getauxval)
    return !!(getauxval(AT_HWCAP) & HWCAP_CRC32);
  return hasHardwareCRC32ARMPosix();
}
#else
bool hasHardwareCRC32() { return false; }
#endif  // defined(__x86_64__) || defined(__i386__)

}  // namespace __scudo
