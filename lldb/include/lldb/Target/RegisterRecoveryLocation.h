//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_REGISTERRECOVERYLOCATION_H
#define LLDB_TARGET_REGISTERRECOVERYLOCATION_H

#include "lldb/lldb-types.h"
#include <cstdint>

namespace lldb_private {

/// A resolved register value source for frame recovery. Unlike
/// RegisterLocation, this identifies a context register, process memory, or an
/// inferred value, rather than pieces of a saved-state bank. The caller
/// performs memory access.
struct RegisterRecoveryLocation {
  enum RegisterLocationTypes {
    eRegisterNotSaved = 0,          // register was not preserved by callee.  If
                                    // volatile reg, is unavailable
    eRegisterSavedAtMemoryLocation, // register is saved at a specific word of
                                    // target mem (target_memory_location)
    eRegisterInRegister, // register is available in a (possible other)
                         // register (register_number)
    eRegisterIsRegisterPlusOffset, // register is available in a (possible
                                   // other) register (register_number) with
                                   // an offset applied
    eRegisterSavedAtHostMemoryLocation, // register is saved at a word in
                                        // lldb's address space
    eRegisterValueInferred,             // register val was computed (and is in
                                        // inferred_value)
    eRegisterInLiveRegisterContext // register value is in a live (stack frame
                                   // #0) register
  };
  int type;
  union {
    lldb::addr_t target_memory_location;
    uint32_t register_number; // in eRegisterKindLLDB register numbering system
    void *host_memory_location;
    uint64_t inferred_value; // eRegisterValueInferred - e.g. stack pointer ==
                             // cfa + offset
    struct {
      uint32_t
          register_number; // in eRegisterKindLLDB register numbering system
      uint64_t offset;
    } reg_plus_offset;
  } location;
};

} // namespace lldb_private

#endif
