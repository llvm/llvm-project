//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared ABI contract for GPU sanitizer reports shipped to the host over RPC.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_CSAN_INTERFACE_H
#define SANITIZER_CSAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

// RPC opcode identifying a GPU race report. The high byte tags the sanitizer
// family ('s') so distinct GPU tools can share the RPC channel.
static const unsigned TSAN_GPU_REPORT_OPCODE = ('s' << 24) | 1;

// Access classification flags, mirroring the Linux kernel's KCSAN model.
static const unsigned TSAN_GPU_ACCESS_WRITE = 1 << 0;    // Non-atomic write.
static const unsigned TSAN_GPU_ACCESS_COMPOUND = 1 << 1; // Read-modify-write.
static const unsigned TSAN_GPU_ACCESS_ATOMIC = 1 << 2;   // Atomic access.

// Race report kinds.
static const unsigned TSAN_GPU_DATA_RACE = 0;      // Conflicting access.
static const unsigned TSAN_GPU_UNKNOWN_ORIGIN = 1; // Value change.
static const unsigned TSAN_GPU_INTRA_WAVE = 2;     // Race within a wavefront.

// The data associated with a single detected race.
struct __tsan_gpu_race {
  uint64_t pc;             // Device PC of the reporting access.
  uint64_t peer_pc;        // Device PC of the conflicting access.
  uint64_t addr;           // Accessed device address.
  uint32_t size;           // Access size in bytes.
  uint32_t access_type;    // Bitwise-or of TSAN_GPU_ACCESS_* flags.
  uint32_t kind;           // One of the TSAN_GPU_* race kinds.
  uint32_t block[3];       // Block / workgroup id.
  uint16_t thread[3];      // Thread / work-item id within the block.
  uint8_t lane;            // Lane id within the wave.
  uint8_t peer_lane;       // Conflicting lane for intra-wave races.
  uint16_t peer_thread[3]; // Conflicting thread id for intra-wave races.
};

#endif // SANITIZER_CSAN_INTERFACE_H
