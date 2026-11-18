//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// LLDB register IDs for riscv32.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_ARCH_LLDB_RISCV32_REGISTER_ENUMS_H
#define LLDB_REGISTER_ARCH_LLDB_RISCV32_REGISTER_ENUMS_H

#include "lldb/Register/Arch/lldb-riscv-register-enums.h"

namespace lldb_private::registers::riscv32 {

namespace id = lldb_private::registers::riscv::id;

} // namespace lldb_private::registers::riscv32

namespace lldb_private::registers::riscv32::darwin {

// Register IDs for Mach thread-state fields outside the CPU catalog.
enum : uint32_t {
  exc_exception = id::csr_last + 1,
  exc_fsr,
  exc_far,
  csr_bank,
};

} // namespace lldb_private::registers::riscv32::darwin

#endif
