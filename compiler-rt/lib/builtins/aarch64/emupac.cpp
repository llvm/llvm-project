//===--- emupac.cpp - Emulated PAC implementation -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements Emulated PAC using SipHash_1_3 as the IMPDEF hashing
//  scheme.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "siphash/SipHash.h"

// EmuPAC implements runtime emulation of PAC instructions. If the current
// CPU supports PAC, EmuPAC uses real PAC instructions. Otherwise, it uses the
// emulation, which is effectively an implementation of PAC with an IMPDEF
// hashing scheme based on SipHash_1_3.
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
// EmuPAC should not be used in production and is only intended for testing use
// cases. This is not only because of the performance costs, which will exist
// even on PAC-supporting machines because of the function call overhead for
// each sign/auth operation, but because it provides weaker security compared to
// real PAC: the key is constant and public, which means that we do not mix a
// global secret.
//
// The emulation assumes that the VA size is at most 48 bits. The architecture
// as of ARMv8.2, which was the last architecture version in which PAC was not
// mandatory, permitted VA size up to 52 bits via ARMv8.2-LVA, but we are
// unaware of an ARMv8.2 CPU that implemented ARMv8.2-LVA.

static const uint64_t max_va_size = 48;
static const uint64_t pac_mask =
    ((1ULL << 55) - 1) & ~((1ULL << max_va_size) - 1);
static const uint64_t ttbr1_mask = 1ULL << 55;

// Determine whether PAC is supported without accessing memory. This utilizes
// the XPACLRI instruction which will copy bit 55 of x30 into at least bit 54 if
// PAC is supported and acts as a NOP if PAC is not supported.
static bool pac_supported() {
  register uintptr_t x30 __asm__("x30") = 1ULL << 55;
  __asm__ __volatile__("xpaclri" : "+r"(x30));
  return x30 & (1ULL << 54);
}

#ifdef __GCC_HAVE_DWARF2_CFI_ASM
#define CFI_INST(inst) inst
#else
#define CFI_INST(inst)
#endif

#ifdef __APPLE__
#define ASM_SYMBOL(symbol) "_" #symbol
#else
#define ASM_SYMBOL(symbol) #symbol
#endif

// This asm snippet is used to force the creation of a frame record when
// calling the EmuPAC functions. This is important because the EmuPAC functions
// may crash if an auth failure is detected and may be unwound past using a
// frame pointer based unwinder.
// clang-format off
#define FRAME_POINTER_WRAP(sym) \
  CFI_INST(".cfi_startproc\n") \
  "stp x29, x30, [sp, #-16]!\n" \
  CFI_INST(".cfi_def_cfa_offset 16\n") \
  "mov x29, sp\n" \
  CFI_INST(".cfi_def_cfa w29, 16\n") \
  CFI_INST(".cfi_offset w30, -8\n") \
  CFI_INST(".cfi_offset w29, -16\n") \
  "bl " ASM_SYMBOL(sym) "\n" \
  CFI_INST(".cfi_def_cfa wsp, 16\n") \
  "ldp x29, x30, [sp], #16\n" \
  CFI_INST(".cfi_def_cfa_offset 0\n") \
  CFI_INST(".cfi_restore w30\n") \
  CFI_INST(".cfi_restore w29\n") \
  "ret\n" \
  CFI_INST(".cfi_endproc\n")
// clang-format on

// Emulated DA key value.
static const uint8_t emu_da_key[16] = {0xb5, 0xd4, 0xc9, 0xeb, 0x79, 0x10,
                                       0x4a, 0x79, 0x6f, 0xec, 0x8b, 0x1b,
                                       0x42, 0x87, 0x81, 0xd4};

extern "C" [[gnu::flatten]] uint64_t __emupac_pacda_impl(uint64_t ptr,
                                                         uint64_t disc) {
  if (pac_supported()) {
    __asm__ __volatile__(".arch_extension pauth\npacda %0, %1"
                         : "+r"(ptr)
                         : "r"(disc));
    return ptr;
  }
  if (ptr & ttbr1_mask) {
    if ((ptr & pac_mask) != pac_mask) {
      return ptr | pac_mask;
    }
  } else {
    if (ptr & pac_mask) {
      return ptr & ~pac_mask;
    }
  }
  uint64_t hash;
  siphash<1, 3>(reinterpret_cast<uint8_t *>(&ptr), 8, emu_da_key,
                *reinterpret_cast<uint8_t (*)[8]>(&hash));
  return (ptr & ~pac_mask) | (hash & pac_mask);
}

// clang-format off
__asm__(
  ".globl " ASM_SYMBOL(__emupac_pacda) "\n"
  ASM_SYMBOL(__emupac_pacda) ":\n"
  FRAME_POINTER_WRAP(__emupac_pacda_impl)
);
// clang-format on

extern "C" [[gnu::flatten]] uint64_t __emupac_autda_impl(uint64_t ptr,
                                                         uint64_t disc) {
  if (pac_supported()) {
    __asm__ __volatile__(".arch_extension pauth\nautda %0, %1"
                         : "+r"(ptr)
                         : "r"(disc));
    return ptr;
  }
  uint64_t ptr_without_pac =
      (ptr & ttbr1_mask) ? (ptr | pac_mask) : (ptr & ~pac_mask);
  uint64_t hash;
  siphash<1, 3>(reinterpret_cast<uint8_t *>(&ptr_without_pac), 8, emu_da_key,
                *reinterpret_cast<uint8_t (*)[8]>(&hash));
  if (((ptr & ~pac_mask) | (hash & pac_mask)) != ptr) {
    __builtin_trap();
  }
  return ptr_without_pac;
}

// clang-format off
__asm__(
  ".globl " ASM_SYMBOL(__emupac_autda) "\n"
  ASM_SYMBOL(__emupac_autda) ":\n"
  FRAME_POINTER_WRAP(__emupac_autda_impl)
);
// clang-format on
