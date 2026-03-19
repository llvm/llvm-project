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
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <memory>

using namespace llvm;

constexpr static const char *KFD_SYSFS_NODE_PATH =
    "/sys/devices/virtual/kfd/kfd/topology/nodes";

// See the ROCm implementation for how this is handled.
// https://github.com/ROCm/ROCT-Thunk-Interface/blob/master/src/libhsakmt.h#L126
constexpr static long getMajor(long Ver) { return (Ver / 10000) % 100; }
constexpr static long getMinor(long Ver) { return (Ver / 100) % 100; }
constexpr static long getStep(long Ver) { return Ver % 100; }

int printGPUsByKFD() {
  SmallVector<std::pair<long, long>> Devices;
  std::error_code EC;
  for (sys::fs::directory_iterator Begin(KFD_SYSFS_NODE_PATH, EC), End;
       Begin != End; Begin.increment(EC)) {
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
    for (line_iterator Lines(**BufferOrErr, false); !Lines.is_at_end();
         ++Lines) {
      StringRef Line(*Lines);
      if (Line.consume_front("gfx_target_version")) {
        if (Line.drop_while([](char C) { return std::isspace(C); })
                .consumeInteger(10, GFXVersion))
          return 1;
        break;
      }
    }

    // If this is zero the node is a CPU.
    if (GFXVersion == 0)
      continue;
    Devices.emplace_back(Node, GFXVersion);
  }

  // Sort the devices by their node to make sure it prints in order.
  llvm::sort(Devices, [](auto &L, auto &R) { return L.first < R.first; });
  for (const auto &[Node, GFXVersion] : Devices)
    std::fprintf(stdout, "gfx%ld%ld%lx\n", getMajor(GFXVersion),
                 getMinor(GFXVersion), getStep(GFXVersion));

  return 0;
}
