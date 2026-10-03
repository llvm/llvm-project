//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the startup code for bare-metal Hexagon targets.
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

namespace LIBC_NAMESPACE_DECL {

[[noreturn]] void do_start() {
  // TODO: This startup code is the MVP for running under the Hexagon
  // simulator with semihosting. It does not configure the MMU or caches.

  // Clear the Supervisor Status Register so that a subsequent trap0
  // semihosting call is serviced by the monitor.
  asm volatile("ssr = %0" : : "r"(0));

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
  // Setup event vector base (needed for semihosting trap handling).
  asm volatile("r0 = ##__llvm_libc_hexagon_event_vectors\n\t"
               "evb = r0");
  // Setup stack pointer, 16-byte aligned.
  asm volatile("r0 = ##__stack\n\t"
               "sp = and(r0, #-16)");
  // Setup GP register for the small-data area.
  asm volatile("r0 = ##_SDA_BASE_\n\t"
               "gp = r0");
  asm volatile("jump %0" : : "X"(LIBC_NAMESPACE::do_start));
}
} // extern "C"

// The simulator services semihosting when trap0 is executed. Put the trap0
// handler at event-vector slot 8; other slots are unexpected and spin.
asm(".text\n\t"
    ".p2align 4\n"
    "__llvm_libc_hexagon_event_spin:\n\t"
    "jump __llvm_libc_hexagon_event_spin\n"
    "__llvm_libc_hexagon_event_trap0:\n\t"
    "rte\n\t"
    ".p2align 12, 0\n\t"
    ".global __llvm_libc_hexagon_event_vectors\n"
    "__llvm_libc_hexagon_event_vectors:\n\t"
    "jump __llvm_libc_hexagon_event_spin\n\t"  // 0  reset
    "jump __llvm_libc_hexagon_event_spin\n\t"  // 1  nmi
    "jump __llvm_libc_hexagon_event_spin\n\t"  // 2  error
    "jump __llvm_libc_hexagon_event_spin\n\t"  // 3  reserved
    "jump __llvm_libc_hexagon_event_spin\n\t"  // 4  tlb miss x
    "jump __llvm_libc_hexagon_event_spin\n\t"  // 5  reserved
    "jump __llvm_libc_hexagon_event_spin\n\t"  // 6  tlb miss rw
    "jump __llvm_libc_hexagon_event_spin\n\t"  // 7  reserved
    "jump __llvm_libc_hexagon_event_trap0\n\t" // 8  trap0
    "jump __llvm_libc_hexagon_event_spin\n\t"  // 9  trap1
);
