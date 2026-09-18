//===-- Implementation of crt for riscv -----------------------------------===//
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
[[gnu::aligned(4)]] void trap_handler() {
  LIBC_NAMESPACE::exit(1);
}
} // namespace

namespace LIBC_NAMESPACE_DECL {

[[noreturn]] void do_start() {
  // Set up trap handling.
  __asm__ volatile("csrw mtvec, %0" :: "r"(&trap_handler));

#ifdef __riscv_flen
  // Enable FPU by setting FS bits (14:13) to Initial (01) in mstatus.
  __asm__ volatile("csrs mstatus, %0" :: "r"(1UL << 13) : "memory");
  // Clear fcsr.
  __asm__ volatile("fscsr zero");
#endif
#ifdef __riscv_vector
  // Enable Vector unit by setting VS bits (10:9) to Initial (01) in mstatus.
  __asm__ volatile("csrs mstatus, %0" :: "r"(1UL << 9) : "memory");
#endif

  LIBC_NAMESPACE::memcpy(__data_start, __data_source,
                         reinterpret_cast<uintptr_t>(__data_size));
  LIBC_NAMESPACE::memset(__bss_start, '\0',
                         reinterpret_cast<uintptr_t>(__bss_size));
  __libc_init_array();

  _platform_init();
  LIBC_NAMESPACE::atexit(&__libc_fini_array);
  LIBC_NAMESPACE::exit(main(0, nullptr));
}
} // namespace LIBC_NAMESPACE_DECL

extern "C" {
[[gnu::section(".text.init.enter"), gnu::naked]]
void _start() {
  // Initialize global pointer if defined by linker.
  // We use .option norelax to prevent the assembler from relaxing the lla instruction.
  __asm__ volatile(
      ".option push\n"
      ".option norelax\n"
      "lla gp, __global_pointer$\n"
      ".option pop\n"
  );

  // Initialize stack pointer.
  __asm__ volatile("lla sp, %0" : : "i"(&__stack));

  // Call do_start.
  __asm__ volatile("tail %0" : : "i"(&LIBC_NAMESPACE::do_start));
}
} // extern "C"
