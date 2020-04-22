//===-- xray_interface_internal.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Implementation of the API functions. See also include/xray/xray_interface.h.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_INTERFACE_INTERNAL_H
#define XRAY_INTERFACE_INTERNAL_H

#include "sanitizer_common/sanitizer_platform.h"
#include "xray/xray_interface.h"
#include <cstddef>
#include <cstdint>

extern "C" {

struct XRaySledEntry {
#if SANITIZER_WORDSIZE == 64
  uint64_t Address;
  uint64_t Function;
  unsigned char Kind;
  unsigned char AlwaysInstrument;
  unsigned char Version;
  unsigned char Padding[13]; // Need 32 bytes
  uint64_t address() const {
#ifndef __x86_64__
    // R_MIPS_PC64 does not exist. Use absolute address even for version 2.
    return Address;
#else
    // TODO Eventually all targets but MIPS64 should take this branch.
    if (Version < 2)
      return Address;
    // The target address is relative to the location of the Address variable.
    return reinterpret_cast<uint64_t>(&Address) + Address;
#endif
  }
#elif SANITIZER_WORDSIZE == 32
  uint32_t Address;
  uint32_t Function;
  unsigned char Kind;
  unsigned char AlwaysInstrument;
  unsigned char Version;
  unsigned char Padding[5]; // Need 16 bytes
  uint32_t address() const { return Address; }
#else
#error "Unsupported word size."
#endif
};

struct XRayFunctionSledIndex {
  const XRaySledEntry *Begin;
  const XRaySledEntry *End;
};
}

namespace __xray {

struct XRaySledMap {
  const XRaySledEntry *Sleds;
  size_t Entries;
  const XRayFunctionSledIndex *SledsIndex;
  size_t Functions;
};

bool patchFunctionEntry(bool Enable, uint32_t FuncId, const XRaySledEntry &Sled,
                        void (*Trampoline)());
bool patchFunctionExit(bool Enable, uint32_t FuncId, const XRaySledEntry &Sled);
bool patchFunctionTailExit(bool Enable, uint32_t FuncId,
                           const XRaySledEntry &Sled);
bool patchCustomEvent(bool Enable, uint32_t FuncId, const XRaySledEntry &Sled);
bool patchTypedEvent(bool Enable, uint32_t FuncId, const XRaySledEntry &Sled);

} // namespace __xray

extern "C" {
// The following functions have to be defined in assembler, on a per-platform
// basis. See xray_trampoline_*.S files for implementations.
extern void __xray_FunctionEntry();
extern void __xray_FunctionExit();
extern void __xray_FunctionTailExit();
extern void __xray_ArgLoggerEntry();
extern void __xray_CustomEvent();
extern void __xray_TypedEvent();
}

#endif
