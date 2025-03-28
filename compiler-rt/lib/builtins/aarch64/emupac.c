#include <stdint.h>

#define XXH_INLINE_ALL
#define XXH_NO_STDLIB
#define XXH_memcpy __builtin_memcpy
#define XXH_memset __builtin_memset
#define XXH_memcmp __builtin_memcmp
#include "../xxhash.h"

// EmuPAC implements runtime emulation of PAC instructions. If the current
// CPU supports PAC, EmuPAC uses real PAC instructions. Otherwise, it uses the
// emulation, which is effectively an implementation of PAC with an IMPDEF
// hashing scheme based on XXH128.
//
// The purpose of the emulation is to allow programs to be built to be portable
// to machines without PAC support, with some performance loss and increased
// probability of false positives (due to not being able to portably determine
// the VA size), while being functionally almost equivalent to running on a
// machine with PAC support. One example of a use case is if PAC is used in
// production as a security mitigation, but the testing environment is
// heterogeneous (i.e. some machines lack PAC support). In this case we would
// like the testing machines to be able to detect issues resulting
// from the use of PAC instructions that would affect production by running
// tests. This can be achieved by building test binaries with EmuPAC and
// production binaries with real PAC.
//
// The emulation assumes that the VA size is at most 48 bits. The architecture
// as of ARMv8.2, which was the last architecture version in which PAC was not
// mandatory, permitted VA size up to 52 bits via ARMv8.2-LVA, but we are
// unaware of an ARMv8.2 CPU that implemented ARMv8.2-LVA.

const uint64_t kMaxVASize = 48;
const uint64_t kPACMask = ((1ULL << 55) - 1) & ~((1ULL << kMaxVASize) - 1);
const uint64_t kTTBR1Mask = 1ULL << 55;

// Determine whether PAC is supported without accessing memory. This utilizes
// the XPACLRI instruction which will copy bit 55 of x30 into at least bit 54 if
// PAC is supported and acts as a NOP if PAC is not supported.
static _Bool pac_supported() {
  register uintptr_t x30 __asm__("x30") = 1ULL << 55;
  __asm__ __volatile__("xpaclri" : "+r"(x30));
  return x30 & (1ULL << 54);
}

// This asm snippet is used to force the creation of a frame record when
// calling the EmuPAC functions. This is important because the EmuPAC functions
// may crash if an auth failure is detected and may be unwound past using a
// frame pointer based unwinder.
#ifdef __GCC_HAVE_DWARF2_CFI_ASM
#define frame_pointer_wrap(sym) \
  "stp x29, x30, [sp, #-16]!\n" \
  ".cfi_def_cfa_offset 16\n" \
  "mov x29, sp\n" \
  ".cfi_def_cfa w29, 16\n" \
  ".cfi_offset w30, -8\n" \
  ".cfi_offset w29, -16\n" \
  "bl " #sym "\n" \
  ".cfi_def_cfa wsp, 16\n" \
  "ldp x29, x30, [sp], #16\n" \
  ".cfi_def_cfa_offset 0\n" \
  ".cfi_restore w30\n" \
  ".cfi_restore w29\n" \
  "ret"
#else
#define frame_pointer_wrap(sym) \
  "stp x29, x30, [sp, #-16]!\n" \
  "mov x29, sp\n" \
  "bl " #sym "\n" \
  "ldp x29, x30, [sp], #16\n" \
  "ret"
#endif

uint64_t __emupac_pacda_impl(uint64_t ptr, uint64_t disc) {
  if (pac_supported()) {
    __asm__ __volatile__(".arch_extension pauth\npacda %0, %1"
                         : "+r"(ptr)
                         : "r"(disc));
    return ptr;
  }
  if (ptr & kTTBR1Mask) {
    if ((ptr & kPACMask) != kPACMask) {
      return ptr | kPACMask;
    }
  } else {
    if (ptr & kPACMask) {
      return ptr & ~kPACMask;
    }
  }
  uint64_t hash = XXH3_64bits_withSeed(&ptr, 8, disc);
  return (ptr & ~kPACMask) | (hash & kPACMask);
}

__attribute__((naked)) uint64_t __emupac_pacda(uint64_t ptr, uint64_t disc) {
  __asm__(frame_pointer_wrap(__emupac_pacda_impl));
}

uint64_t __emupac_autda_impl(uint64_t ptr, uint64_t disc) {
  if (pac_supported()) {
    __asm__ __volatile__(".arch_extension pauth\nautda %0, %1"
                         : "+r"(ptr)
                         : "r"(disc));
    return ptr;
  }
  uint64_t ptr_without_pac =
      (ptr & kTTBR1Mask) ? (ptr | kPACMask) : (ptr & ~kPACMask);
  uint64_t hash = XXH3_64bits_withSeed(&ptr_without_pac, 8, disc);
  if (((ptr & ~kPACMask) | (hash & kPACMask)) != ptr) {
    __builtin_trap();
  }
  return ptr_without_pac;
}

__attribute__((naked)) uint64_t __emupac_autda(uint64_t ptr, uint64_t disc) {
  __asm__(frame_pointer_wrap(__emupac_autda_impl));
}
