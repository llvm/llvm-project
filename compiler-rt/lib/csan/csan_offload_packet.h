//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// RPC packet shared by the device and host ConcurrencySanitizer runtimes.
///
//===----------------------------------------------------------------------===//

#ifndef CSAN_OFFLOAD_PACKET_H
#define CSAN_OFFLOAD_PACKET_H

#include "csan_defs.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_offload_opcodes.h"

struct __csan_gpu_race {
  __sanitizer::u64 pc;
  __sanitizer::u64 peer_pc;
  __sanitizer::u64 addr;
  __sanitizer::u32 size;
  __sanitizer::u32 access_type;
  __sanitizer::u32 kind;
  __sanitizer::u32 block[3];
  __sanitizer::u16 thread[3];
  __sanitizer::u8 lane;
  __sanitizer::u8 peer_lane;
  __sanitizer::u8 peer_access_type;
  __sanitizer::u8 peer_size;
};

static_assert(sizeof(__csan_gpu_race) == 64,
              "Offload CSan report must fit one RPC packet");

#endif // CSAN_OFFLOAD_PACKET_H
