//===-- Implementation of crt for arm -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"
#include "src/string/memcpy.h"
#include "src/string/memset.h"
#include "startup/baremetal/fini.h"
#include "startup/baremetal/init.h"

#include <arm_acle.h> // For __arm_wsr

extern "C" {
int main(int argc, char **argv);
void _start();

// Semihosting library initialisation if applicable. Required for printf, etc.
[[gnu::weak]] void _platform_init() {}

// These symbols are provided by the linker. The exact names are not defined by
// a standard.
extern uintptr_t __stack;
extern uintptr_t __data_source[];
extern uintptr_t __data_start[];
extern uintptr_t __data_size[];
extern uintptr_t __bss_start[];
extern uintptr_t __bss_size[];
} // extern "C"

namespace {
#if __ARM_ARCH_PROFILE == 'A' && !defined(__ARM_ARCH_ISA_A64) && __ARM_ARCH >= 7
constexpr uint32_t PAGE_TABLE_ENTRY_COUNT = 4096;
constexpr uint32_t PAGE_TABLE_ALIGNMENT = 16384;

// Put the page table in a no-init section so it doesn't later get
// zero-initialized.
[[gnu::section(".noinit.page_table"), gnu::aligned(PAGE_TABLE_ALIGNMENT),
  gnu::used]] volatile uint32_t page_table[PAGE_TABLE_ENTRY_COUNT];

void setup_mmu() {
  constexpr uint32_t PAGE_SHIFT = 20;

  // Fill the page table with a flat mapping of 4096 1MB sections with all
  // sections marked as normal.
  //  base address = bits 20:31
  //  bits 18:19 set to 0
  //  nG = bit 17 set to 0 (global)
  //  S = bit 16 set to 0 (non-shared)
  //  APX = bit 15 set to 0 (full read/write)
  //  TEX = bits 12:14 = b111 (normal)
  //  AP = bits 10:11 set to b11 (full read/write)
  //  P = bit 9 set to 0 (no ECC)
  //  domain = bits 5:8 = b000
  //  XN = bit 4 set to 0
  //  C, B bits = bits 2:3 set to b11 (normal)
  //  size = 1MB = bits 0:1 set to b10
  constexpr uint32_t PAGE_TABLE_ENTRY = 0x7c0e;

  uint32_t value = 3;
  __arm_wsr("p15:0:c3:c0:0", value); // DACR: manager access to domain 0.
  value = 0;
  __arm_wsr("p15:0:c2:c0:2", value); // TTBCR: always use TTBR0.
  value = reinterpret_cast<uint32_t>(page_table) | 1;
  __arm_wsr("p15:0:c2:c0:0", value); // TTBR0: inner-cacheable walks.
  __isb(0xF);

  for (uint32_t page = 0; page < PAGE_TABLE_ENTRY_COUNT; ++page)
    page_table[page] = PAGE_TABLE_ENTRY | (page << PAGE_SHIFT);

  __dsb(0xF);

  uint32_t sctlr = __arm_rsr("p15:0:c1:c0:0");
#ifdef __ARM_FEATURE_UNALIGNED
  sctlr &= ~(1 << 1); // SCTLR.A: disable alignment checks.
  sctlr |= 1 << 22;   // SCTLR.U: enable unaligned access support.
#else
  sctlr |= 1 << 1; // SCTLR.A: enable alignment checks.
#endif
  sctlr |= 1 << 0;  // SCTLR.M: enable MMU.
  sctlr |= 1 << 2;  // SCTLR.C: enable data cache.
  sctlr |= 1 << 12; // SCTLR.I: enable instruction cache.
  __arm_wsr("p15:0:c1:c0:0", sctlr);
  __isb(0xF);
}
#endif

#if __ARM_ARCH_PROFILE == 'M'
// Based on
// https://developer.arm.com/documentation/107565/0101/Use-case-examples/Generic-Information/What-is-inside-a-program-image-/Vector-table
void NMI_Handler() {}
void HardFault_Handler() { LIBC_NAMESPACE::exit(1); }
void MemManage_Handler() { LIBC_NAMESPACE::exit(1); }
void BusFault_Handler() { LIBC_NAMESPACE::exit(1); }
void UsageFault_Handler() { LIBC_NAMESPACE::exit(1); }
void SVC_Handler() {}
void DebugMon_Handler() {}
void PendSV_Handler() {}
void SysTick_Handler() {}

// Architecturally the bottom 7 bits of VTOR are zero, meaning the vector table
// has to be 128-byte aligned, however an implementation can require more bits
// to be zero and Cortex-M23 can require up to 10, so 1024-byte align the vector
// table.
using HandlerType = void (*)(void);
[[gnu::section(".vectors"), gnu::aligned(1024), gnu::used]]
const HandlerType vector_table[] = {
    reinterpret_cast<HandlerType>(&__stack), // SP
    _start,                                  // Reset
    NMI_Handler,                             // NMI Handler
    HardFault_Handler,                       // Hard Fault Handler
    MemManage_Handler,                       // MPU Fault Handler
    BusFault_Handler,                        // Bus Fault Handler
    UsageFault_Handler,                      // Usage Fault Handler
    0,                                       // Reserved
    0,                                       // Reserved
    0,                                       // Reserved
    0,                                       // Reserved
    SVC_Handler,                             // SVC Handler
    DebugMon_Handler,                        // Debug Monitor Handler
    0,                                       // Reserved
    PendSV_Handler,                          // PendSV Handler
    SysTick_Handler,                         // SysTick Handler
                                             // Unused
};
#else
// Based on
// https://developer.arm.com/documentation/den0013/0400/Boot-Code/Booting-a-bare-metal-system
void Reset_Handler() { LIBC_NAMESPACE::exit(1); }
void Undefined_Handler() { LIBC_NAMESPACE::exit(1); }
void SWI_Handler() { LIBC_NAMESPACE::exit(1); }
void PrefetchAbort_Handler() { LIBC_NAMESPACE::exit(1); }
void DataAbort_Handler() { LIBC_NAMESPACE::exit(1); }
void IRQ_Handler() { LIBC_NAMESPACE::exit(1); }
void FIQ_Handler() { LIBC_NAMESPACE::exit(1); }

// The AArch32 exception vector table has 8 entries, each of which is 4
// bytes long, and contains code. The whole table must be 32-byte aligned.
// The table may also be relocated, so we make it position-independent by
// having a table of handler addresses and loading the address to pc.
[[gnu::section(".vectors"), gnu::aligned(32), gnu::used, gnu::naked,
  gnu::target("arm")]]
void vector_table() {
  asm("LDR pc, [pc, #24]");
  asm("LDR pc, [pc, #24]");
  asm("LDR pc, [pc, #24]");
  asm("LDR pc, [pc, #24]");
  asm("LDR pc, [pc, #24]");
  asm("LDR pc, [pc, #24]");
  asm("LDR pc, [pc, #24]");
  asm("LDR pc, [pc, #24]");
  asm(".word %c0" : : "X"(Reset_Handler));
  asm(".word %c0" : : "X"(Undefined_Handler));
  asm(".word %c0" : : "X"(SWI_Handler));
  asm(".word %c0" : : "X"(PrefetchAbort_Handler));
  asm(".word %c0" : : "X"(DataAbort_Handler));
  asm(".word %c0" : : "X"(0));
  asm(".word %c0" : : "X"(IRQ_Handler));
  asm(".word %c0" : : "X"(FIQ_Handler));
}
#endif
} // namespace

namespace LIBC_NAMESPACE_DECL {
[[noreturn]] void do_start() {
  // FIXME: set up the QEMU test environment

#if __ARM_ARCH_PROFILE == 'A' || __ARM_ARCH_PROFILE == 'R'
  // Set up registers to be used in exception handling
  // Copy the current sp value to each of the banked copies of sp.
  asm volatile("mov r0, sp\n"
               "mov r1, #0x11\n" // FIQ
               "msr CPSR_c, r1\n"
               "mov sp, r0\n"
               "mov r1, #0x12\n" // IRQ
               "msr CPSR_c, r1\n"
               "mov sp, r0\n"
               "mov r1, #0x17\n" // ABT
               "msr CPSR_c, r1\n"
               "mov sp, r0\n"
               "mov r1, #0x1B\n" // UND
               "msr CPSR_c, r1\n"
               "mov sp, r0\n"
               "mov r1, #0x1F\n" // SYS
               "msr CPSR_c, r1\n"
               "mov sp, r0\n"
               "mov r1, #0x13\n" // return to SVC
               "msr CPSR_c, r1"
               :
               :
               : "r0", "r1");
#endif

#if __ARM_ARCH_PROFILE == 'A' && !defined(__ARM_ARCH_ISA_A64) && __ARM_ARCH >= 7
  __arm_wsr("p15:0:c12:c0:0", reinterpret_cast<uint32_t>(&vector_table));
  setup_mmu();
#endif

#if __ARM_ARCH_PROFILE == 'M' && !defined(__ARM_FEATURE_UNALIGNED)
  auto ccr = reinterpret_cast<volatile uint32_t *const>(0xE000ED14);
  *ccr |= 1 << 3; // CCR.UNALIGN_TRP: trap unaligned accesses.
#endif

#if __ARM_ARCH_PROFILE == 'M' &&                                               \
    (defined(__ARM_FP) || defined(__ARM_FEATURE_MVE))
  // Enable FPU and MVE. They can't be enabled independently: the two are
  // governed by the same bits in CPACR.
  // Based on
  // https://developer.arm.com/documentation/dui0646/c/Cortex-M7-Peripherals/Floating-Point-Unit/Enabling-the-FPU
  // Set CPACR cp10 and cp11.
  auto cpacr = reinterpret_cast<volatile uint32_t *const>(0xE000ED88);
  *cpacr |= (0xF << 20);
  __dsb(0xF);
  __isb(0xF);
#if defined(__ARM_FEATURE_MVE)
  // Initialize low-overhead-loop tail predication to its neutral state
  uint32_t fpscr;
  __asm__ __volatile__("vmrs %0, FPSCR" : "=r"(fpscr) : :);
  fpscr |= (0x4 << 16);
  __asm__ __volatile__("vmsr FPSCR, %0" : : "r"(fpscr) :);
#endif
#elif (__ARM_ARCH_PROFILE == 'A' || __ARM_ARCH_PROFILE == 'R') &&              \
    defined(__ARM_FP)
  // Enable FPU.
  // Based on
  // https://developer.arm.com/documentation/dui0472/m/Compiler-Coding-Practices/Enabling-NEON-and-FPU-for-bare-metal
  // Set CPACR cp10 and cp11.
  uint32_t cpacr = __arm_rsr("p15:0:c1:c0:2");
  cpacr |= (0xF << 20);
  __arm_wsr("p15:0:c1:c0:2", cpacr);
  __isb(0xF);
  // Set FPEXC.EN
  uint32_t fpexc;
  __asm__ __volatile__("vmrs %0, FPEXC" : "=r"(fpexc) : :);
  fpexc |= (0x1 << 30);
  __asm__ __volatile__("vmsr FPEXC, %0" : : "r"(fpexc) :);
#endif

  // Perform the equivalent of scatterloading
  LIBC_NAMESPACE::memcpy(__data_start, __data_source,
                         reinterpret_cast<uintptr_t>(__data_size));
  LIBC_NAMESPACE::memset(__bss_start, '\0',
                         reinterpret_cast<uintptr_t>(__bss_size));
  __libc_init_array();

  _platform_init();
  LIBC_NAMESPACE::atexit(&__libc_fini_array);
  LIBC_NAMESPACE::exit(main(0, 0));
}
} // namespace LIBC_NAMESPACE_DECL

extern "C" {
#ifdef __ARM_ARCH_ISA_ARM
// If ARM state is supported, it must be used (instead of Thumb)
[[gnu::naked, gnu::target("arm")]]
#endif
void _start() {
  asm volatile("mov sp, %0" : : "r"(&__stack));
  asm volatile("bl %0" : : "X"(LIBC_NAMESPACE::do_start));
}
} // extern "C"
