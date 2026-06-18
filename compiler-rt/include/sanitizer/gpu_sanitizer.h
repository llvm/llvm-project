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

#ifdef __cplusplus
extern "C" {
#endif

#define SANITIZER_GPU_OPCODE(n) (('s' << 24) | (n))

#define UBSAN_GPU_REPORT_OPCODE SANITIZER_GPU_OPCODE(0)

/// Longest minimal handler kind ("function-type-mismatch") plus a NUL, sized so
/// the report is one 64-byte RPC packet.
#define UBSAN_GPU_KIND_MAX 24

/// A single minimal-runtime UBSan diagnostic, shared verbatim across the
/// device/host RPC boundary. 'pc' is a raw device address resolved on the host.
typedef struct __ubsan_gpu_report {
  unsigned long long pc;         ///< Device PC of the faulting check (caller).
  unsigned block[3];             ///< Block / workgroup id.
  unsigned thread[3];            ///< Thread / work-item id within the block.
  unsigned lane;                 ///< Lane id within the wave.
  char kind[UBSAN_GPU_KIND_MAX]; ///< Null-terminated check name.
} __ubsan_gpu_report;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_GPU_SANITIZER_H
