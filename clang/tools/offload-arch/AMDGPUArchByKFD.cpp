//===- AMDGPUArchByKFD.cpp - list AMDGPU installed ------*- C++ -*---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool for detecting name of AMD GPUs installed in
// system using the Linux sysfs interface for the AMD KFD driver. This file does
// not respect ROCR_VISIBLE_DEVICES like the ROCm environment would.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <tuple>

using namespace llvm;

constexpr static const char *KFD_SYSFS_NODE_PATH =
    "/sys/devices/virtual/kfd/kfd/topology/nodes";
constexpr static long GFX1250_VERSION = 120500;

// See the ROCm implementation for how this is handled.
// https://github.com/ROCm/ROCT-Thunk-Interface/blob/master/src/libhsakmt.h#L126
constexpr static long getMajor(long Ver) { return (Ver / 10000) % 100; }
constexpr static long getMinor(long Ver) { return (Ver / 100) % 100; }
constexpr static long getStep(long Ver) { return Ver % 100; }

// For A0, print gfx1250-strict to match rocminfo
static StringRef getRevisionSuffix(long GFXVersion, long ASICRevision) {
  return (GFXVersion == GFX1250_VERSION && ASICRevision == 0) ? "-strict" : "";
}

// Exposed for testing
int printGPUsByKFD(StringRef NodePath) {
  struct KFDNode {
    long Node;
    long GFXVersion;
    long ASICRevision;
  };

  SmallVector<KFDNode> Devices;
  std::error_code EC;
  sys::fs::directory_iterator Begin(NodePath, EC), End;

  // Fail if the sysfs topology does not exist (e.g., WSL)
  if (EC)
    return 1;

  for (; Begin != End; Begin.increment(EC)) {
    if (EC)
      return 1;

    long Node = 0;
    if (sys::path::stem(Begin->path()).consumeInteger(10, Node))
      return 1;

    SmallString<0> Path(Begin->path());
    sys::path::append(Path, "properties");

    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFileOrSTDIN(Path);
    if (std::error_code EC = BufferOrErr.getError())
      return 1;

    long GFXVersion = 0;
    uint64_t Capability = 0;
    for (line_iterator Lines(**BufferOrErr, false); !Lines.is_at_end();
         ++Lines) {
      StringRef Line(*Lines);
      if (Line.consume_front("gfx_target_version")) {
        if (Line.drop_while([](char C) { return std::isspace(C); })
                .consumeInteger(10, GFXVersion))
          return 1;
        // Differentiate between capability and capability2
      } else if (Line.consume_front("capability") && !Line.starts_with('2')) {
        if (Line.drop_while([](char C) { return std::isspace(C); })
                .consumeInteger(10, Capability))
          return 1;
      }
    }

    // If this is zero the node is a CPU.
    if (GFXVersion == 0)
      continue;
    // ASIC revision is bits 25:22 in capability
    long ASICRevision = (Capability >> 22) & 0xf;
    Devices.push_back({Node, GFXVersion, ASICRevision});
  }

  // Sort the devices by their node to make sure it prints in order.
  llvm::sort(Devices, [](auto &L, auto &R) { return L.Node < R.Node; });
  for (const auto &[Node, GFXVersion, ASICRevision] : Devices) {
    outs() << "gfx" << getMajor(GFXVersion) << getMinor(GFXVersion)
           << format_hex_no_prefix(getStep(GFXVersion), 1)
           << getRevisionSuffix(GFXVersion, ASICRevision) << '\n';
  }

  return 0;
}

int printGPUsByKFD() { return printGPUsByKFD(KFD_SYSFS_NODE_PATH); }
