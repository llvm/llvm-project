//===-- xray_s390x.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Implementation of s390x routines.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_common.h"
#include "xray_defs.h"
#include "xray_interface_internal.h"
#include <cassert>
#include <cstring>

bool __xray::patchFunctionEntry(const bool Enable, const uint32_t FuncId,
                                const XRaySledEntry &Sled,
                                const XRayTrampolines &Trampolines,
                                bool LogArgs) XRAY_NEVER_INSTRUMENT {
  uint32_t *Address = reinterpret_cast<uint32_t *>(Sled.address());
  // TODO: Trampoline addresses are currently inserted at compile-time, using
  //       __xray_FunctionEntry and __xray_FunctionExit only.
  //       To support DSO instrumentation, trampolines have to be written during
  //       patching (see implementation on X86_64, e.g.).
  if (Enable) {
    // The resulting code is:
    //   stmg    %r2, %r15, 16(%r15)
    //   llilf   %2, FuncID
    //   brasl   %r14, __xray_FunctionEntry@GOT
    // The FuncId and the stmg instruction must be written.

    // Write FuncId into llilf.
    Address[2] = FuncId;
    // Write last part of stmg.
    reinterpret_cast<uint16_t *>(Address)[2] = 0x24;
    // Write first part of stmg.
    Address[0] = 0xeb2ff010;
  } else {
    // j +16 instructions.
    Address[0] = 0xa7f4000b;
  }
  return true;
}

bool __xray::patchFunctionExit(
    const bool Enable, const uint32_t FuncId, const XRaySledEntry &Sled,
    const XRayTrampolines &Trampolines) XRAY_NEVER_INSTRUMENT {
  uint32_t *Address = reinterpret_cast<uint32_t *>(Sled.address());
  // TODO: Trampoline addresses are currently inserted at compile-time, using
  //       __xray_FunctionEntry and __xray_FunctionExit only.
  //       To support DSO instrumentation, trampolines have to be written during
  //       patching (see implementation on X86_64, e.g.).
  if (Enable) {
    // The resulting code is:
    //   stmg    %r2, %r15, 24(%r15)
    //   llilf   %2,FuncID
    //   j       __xray_FunctionEntry@GOT
    // The FuncId and the stmg instruction must be written.

    // Write FuncId into llilf.
    Address[2] = FuncId;
    // Write last part of of stmg.
    reinterpret_cast<uint16_t *>(Address)[2] = 0x24;
    // Write first part of stmg.
    Address[0] = 0xeb2ff010;
  } else {
    // br %14 instruction.
    reinterpret_cast<uint16_t *>(Address)[0] = 0x07fe;
  }
  return true;
}

bool __xray::patchFunctionTailExit(
    const bool Enable, const uint32_t FuncId, const XRaySledEntry &Sled,
    const XRayTrampolines &Trampolines) XRAY_NEVER_INSTRUMENT {
  return patchFunctionExit(Enable, FuncId, Sled, Trampolines);
}

bool __xray::patchCustomEvent(const bool Enable, const uint32_t FuncId,
                              const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  // TODO Implement.
  return false;
}

bool __xray::patchTypedEvent(const bool Enable, const uint32_t FuncId,
                             const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  // TODO Implement.
  return false;
}

extern "C" void __xray_ArgLoggerEntry() XRAY_NEVER_INSTRUMENT {
  // TODO this will have to be implemented in the trampoline assembly file.
}

extern "C" void __xray_FunctionTailExit() XRAY_NEVER_INSTRUMENT {
  // For PowerPC, calls to __xray_FunctionEntry and __xray_FunctionExit
  // are statically inserted into the sled. Tail exits are handled like normal
  // function exits. This trampoline is therefore not implemented.
  // This stub is placed here to avoid linking issues.
}
