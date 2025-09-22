//===-- Implementation of crt for aarch64 ---------------------------------===//
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

#include <arm_acle.h>

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
// The Arm ARM for the A-profile architecture (D14.1.5) defines the exceptions.
// However, for simplicity, we don't bother logging, and just exit.
void GenericException_Handler() { LIBC_NAMESPACE::exit(1); }

// The AArch64 exception vector table has 16 entries, each of which is 128
// bytes long, and contains code. The whole table must be 2048-byte aligned.
// For our purposes, each entry just contains one branch instruction to the
// exception reporting function, since we never want to resume after an
// exception.
[[gnu::section(".vectors"), gnu::aligned(2048), gnu::used, gnu::naked]]
void vector_table() {
#define VECTOR_TABLE_ENTRY                                                     \
  asm(".balign 128");                                                          \
  asm("B %0" : : "X"(GenericException_Handler));

  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
  VECTOR_TABLE_ENTRY;
}
} // namespace

namespace LIBC_NAMESPACE_DECL {

[[noreturn]] void do_start() {
  // TODO: This startup code is not extensive, but rather the MVP for QEMU
  // testing.
  // TODO: Setup memory (MMU, page table, caches)
  // TODO: Consider v8-R variants

  // Set up exception handling
  __arm_wsr64("VBAR_EL1", reinterpret_cast<uint64_t>(&vector_table));

#ifdef __ARM_FP
  // Do not trap FP/SME/SVE instructions
  static constexpr uint64_t CPACR_SHIFT_FPEN = 20;
  static constexpr uint64_t CPACR_SHIFT_SMEN = 24;
  uint64_t cpacr = __arm_rsr64("CPACR_EL1");
  cpacr |= (0x3 << CPACR_SHIFT_FPEN);
  cpacr |= (0x3 << CPACR_SHIFT_SMEN);
  __arm_wsr64("CPACR_EL1", cpacr);
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
[[gnu::section(".text.init.enter"), gnu::naked]]
void _start() {
  asm volatile("mov sp, %0" : : "r"(&__stack));
  asm volatile("bl %0" : : "X"(LIBC_NAMESPACE::do_start));
}
} // extern "C"
