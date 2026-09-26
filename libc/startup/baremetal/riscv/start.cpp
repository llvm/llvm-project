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

namespace LIBC_NAMESPACE_DECL {

extern "C" {

int main(int argc, char **argv, char **envp);
[[noreturn]] void _start();
void _platform_init();

// These symbols are provided by the linker script. The exact names are not
// defined by a standard. The _size symbols use the symbol table to store
// integers rather than addresses.

extern uintptr_t __data_source;
extern uintptr_t __data_start;
extern uintptr_t __data_size;

extern uintptr_t __bss_start;
extern uintptr_t __bss_size;

} // extern "C"

// Semihosting library initialization if applicable. Required for printf, etc.
[[gnu::weak]] void _platform_init() {}

namespace {

// Enable Vector unit by setting VS (10:9) bits (01 = Initial).
[[maybe_unused]] constexpr uint32_t MSTATUS_VS_INITIAL = 1U << 9;
// Enable FPU by setting FS (14:13) bits (01 = Initial).
[[maybe_unused]] constexpr uint32_t MSTATUS_FS_INITIAL = 1U << 13;
[[maybe_unused]] constexpr uint32_t MSTATUS_INITIAL =
#ifdef __riscv_flen
    MSTATUS_FS_INITIAL |
#endif
#ifdef __riscv_vector
    MSTATUS_VS_INITIAL |
#endif
    0;

// The mtvec BASE field must be aligned on a 4-byte boundary, with the lower
// two bits holding the MODE (0 = Direct). With the C extension, functions are
// only 2-byte aligned by default.
[[gnu::aligned(4)]] void trap_handler() { LIBC_NAMESPACE::exit(1); }

[[noreturn]] void do_start() {
  // Set up trap handling.
  __asm__ volatile("csrw mtvec, %0" : : "r"(&trap_handler));

  LIBC_NAMESPACE::memcpy(&__data_start, &__data_source,
                         reinterpret_cast<uintptr_t>(&__data_size));
  LIBC_NAMESPACE::memset(&__bss_start, '\0',
                         reinterpret_cast<uintptr_t>(&__bss_size));

  _platform_init();
  __libc_init_array();
  LIBC_NAMESPACE::atexit(&__libc_fini_array);
  LIBC_NAMESPACE::exit(main(0, nullptr, nullptr));
}

} // namespace

[[noreturn, gnu::section(".text.init.enter"), gnu::naked]]
void _start() {
  // Initialize global pointer and stack pointer.
  __asm__(R"(
      .option push
      .option exact
      lla gp, __global_pointer$
      .option pop
      lla sp, __stack
  )");

#ifdef __riscv_zcmt
  // Initialize jvt CSR if defined by linker.
  __asm__(R"(
      .weak __jvt_base$
      lla t0, __jvt_base$
      beqz t0, 1f
      csrw jvt, t0
    1:
  )");
#endif

#if defined(__riscv_flen) || defined(__riscv_vector)
  // Enable FPU and/or Vector unit.
  __asm__(
      R"(
      li t0, %[mstatus_initial]
      csrs mstatus, t0
      )"
      :
      : [mstatus_initial] "i"(MSTATUS_INITIAL));
#endif

  // Call do_start.
  __asm__("tail %cc0" : : "s"(do_start));
}

} // namespace LIBC_NAMESPACE_DECL
