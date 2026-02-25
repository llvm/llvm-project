#include <stdint.h>

#ifdef __aarch64__

static bool pac_supported() {
  register uintptr_t x30 __asm__("x30") = 1ULL << 55;
  __asm__ __volatile__("xpaclri" : "+r"(x30));
  return x30 & (1ULL << 54);
}

static bool fpac_supported(uint64_t ap) {
  // The meaning of values larger than 6 is reserved as of July 2025; in theory
  // larger values could mean that FEAT_FPAC is not implemented.
  return ap == 4 || ap == 5 || ap == 6;
}

#endif

// The crash tests would fail to crash (causing the test to fail) if:
// - The operating system did not enable the DA key, or
// - The CPU supports FEAT_PAuth but not FEAT_FPAC.
// Therefore, they call this function, which will crash the test process if one
// of these cases is detected so that %expect_crash detects the crash and causes
// the test to pass.
//
// We detect the former case by attempting to sign a pointer. If the signed
// pointer is equal to the unsigned pointer, DA is likely disabled, so we crash.
//
// We detect the latter case by reading ID_AA64ISAR1_EL1 and ID_AA64ISAR2_EL1.
// It is expected that the operating system will either trap and emulate reading
// the system registers (as Linux does) or crash the process. In the
// trap/emulate case we check the APA, API and APA3 fields for FEAT_FPAC support
// and crash if it is not available. In the crash case we will crash when
// reading the register leading to a passing test. This means that operating
// systems with the crashing behavior do not support the crash tests.
//
// On non-AArch64 platforms, the generic form of PFP is used which will always
// detect the bug, so this function does nothing.
static void crash_if_crash_tests_unsupported() {
#ifdef __aarch64__
  if (!pac_supported())
    return;

  uint64_t ptr = 0;
  __asm__ __volatile__(".arch_extension pauth\npacda %0, %1"
                       : "+r"(ptr)
                       : "r"(0ul));
  if (ptr == 0)
    __builtin_trap();

  uint64_t aa64isar1;
  __asm__ __volatile__("mrs %0, id_aa64isar1_el1" : "=r"(aa64isar1));
  uint64_t apa = (aa64isar1 >> 4) & 0xf;
  uint64_t api = (aa64isar1 >> 8) & 0xf;
  if (fpac_supported(apa) || fpac_supported(api))
    return;

  uint64_t aa64isar2;
  __asm__ __volatile__("mrs %0, id_aa64isar2_el1" : "=r"(aa64isar2));
  uint64_t apa3 = (aa64isar2 >> 12) & 0xf;
  if (fpac_supported(apa3))
    return;

  __builtin_trap();
#endif
}
