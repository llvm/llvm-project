//===-- xray_interface_internal.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "xray/xray_interface.h"
#include <cstddef>
#include <cstdint>

extern "C" {

struct XRaySledEntry {
  uint64_t Address;
  uint64_t Function;
  unsigned char Kind;
  unsigned char AlwaysInstrument;
  unsigned char Padding[14]; // Need 32 bytes
};
}

namespace __xray {

struct XRaySledMap {
  const XRaySledEntry *Sleds;
  size_t Entries;
};

} // namespace __xray

#endif
