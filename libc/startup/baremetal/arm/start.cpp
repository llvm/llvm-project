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
    HardFault_Handler,                       // Hard Fault Handlerß
    MemManage_Handler,                       // MPU Fault Han`dler
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
} // namespace

namespace LIBC_NAMESPACE_DECL {
[[noreturn]] void do_start() {
  // FIXME: set up the QEMU test environment

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

extern "C" void _start() {
  asm volatile("mov sp, %0" : : "r"(&__stack));
  asm volatile("bl %0" : : "X"(LIBC_NAMESPACE::do_start));
}
