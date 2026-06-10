//===-- sanitizer/gpu_sanitizer.h -------------------------------*- C++ -*-===//
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

#ifndef SANITIZER_GPU_SANITIZER_H
#define SANITIZER_GPU_SANITIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SANITIZER_GPU_OPCODE(n) (('s' << 24) | (n))

#define TSAN_GPU_REPORT_OPCODE SANITIZER_GPU_OPCODE(1)

/// Access classification flags, mirroring the Linux kernel's KCSAN model.
enum {
  TSAN_GPU_ACCESS_WRITE = 1 << 0,    ///< Non-atomic write.
  TSAN_GPU_ACCESS_COMPOUND = 1 << 1, ///< Read-modify-write.
  TSAN_GPU_ACCESS_ATOMIC = 1 << 2,   ///< Atomic access.
};

/// Race report kinds.
enum {
  TSAN_GPU_DATA_RACE = 0,      ///< Conflicting access.
  TSAN_GPU_UNKNOWN_ORIGIN = 1, ///< Watched value changed with no finder.
  TSAN_GPU_INTRA_WAVE = 2,     ///< Conflict between lanes of the same wave.
};

/// The data associated with a single detected race.
typedef struct __tsan_gpu_race {
  uint64_t pc;             ///< Device PC of the reporting access.
  uint64_t peer_pc;        ///< Device PC of the conflicting access.
  uint64_t addr;           ///< Accessed device address.
  uint32_t size;           ///< Access size in bytes.
  uint32_t access_type;    ///< Bitwise-or of TSAN_GPU_ACCESS_* flags.
  uint32_t kind;           ///< One of the TSAN_GPU_* race kinds.
  uint32_t block[3];       ///< Block / workgroup id.
  uint16_t thread[3];      ///< Thread / work-item id within the block.
  uint8_t lane;            ///< Lane id within the wave.
  uint8_t peer_lane;       ///< Conflicting lane for intra-wave races.
  uint16_t peer_thread[3]; ///< Conflicting thread id for intra-wave races.
} __tsan_gpu_race;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_GPU_SANITIZER_H
