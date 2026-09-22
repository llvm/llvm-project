//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RPC opcodes shared by device handlers and the host shim. Packet payloads
// stay with each sanitizer.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_OFFLOAD_OPCODES_H
#define SANITIZER_OFFLOAD_OPCODES_H

#define SANITIZER_OFFLOAD_RPC_BASE 's'
#define SANITIZER_OFFLOAD_OPCODE(n) \
  (((unsigned)(SANITIZER_OFFLOAD_RPC_BASE) << 24) | (unsigned)(n))

enum {
  SANITIZER_OFFLOAD_UBSAN = SANITIZER_OFFLOAD_OPCODE(0),
  SANITIZER_OFFLOAD_CSAN = SANITIZER_OFFLOAD_OPCODE(1),
};

#undef SANITIZER_OFFLOAD_RPC_BASE
#undef SANITIZER_OFFLOAD_OPCODE

#endif  // SANITIZER_OFFLOAD_OPCODES_H
