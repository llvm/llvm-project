//===-- Definition of RPC opcodes -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SHARED_RPC_OPCODES_H
#define LLVM_LIBC_SHARED_RPC_OPCODES_H

#include "rpc.h"

#define LLVM_LIBC_RPC_BASE 'c'
#define LLVM_LIBC_OPCODE(n) (LLVM_LIBC_RPC_BASE << 24 | n)

typedef enum {
  LIBC_NOOP = LLVM_LIBC_OPCODE(0),
  LIBC_EXIT = LLVM_LIBC_OPCODE(1),
  LIBC_WRITE_TO_STDOUT = LLVM_LIBC_OPCODE(2),
  LIBC_WRITE_TO_STDERR = LLVM_LIBC_OPCODE(3),
  LIBC_WRITE_TO_STREAM = LLVM_LIBC_OPCODE(4),
  LIBC_WRITE_TO_STDOUT_NEWLINE = LLVM_LIBC_OPCODE(5),
  LIBC_READ_FROM_STREAM = LLVM_LIBC_OPCODE(6),
  LIBC_READ_FGETS = LLVM_LIBC_OPCODE(7),
  LIBC_OPEN_FILE = LLVM_LIBC_OPCODE(8),
  LIBC_CLOSE_FILE = LLVM_LIBC_OPCODE(9),
  LIBC_MALLOC = LLVM_LIBC_OPCODE(10),
  LIBC_FREE = LLVM_LIBC_OPCODE(11),
  LIBC_HOST_CALL = LLVM_LIBC_OPCODE(12),
  LIBC_ABORT = LLVM_LIBC_OPCODE(13),
  LIBC_FEOF = LLVM_LIBC_OPCODE(14),
  LIBC_FERROR = LLVM_LIBC_OPCODE(15),
  LIBC_CLEARERR = LLVM_LIBC_OPCODE(16),
  LIBC_FSEEK = LLVM_LIBC_OPCODE(17),
  LIBC_FTELL = LLVM_LIBC_OPCODE(18),
  LIBC_FFLUSH = LLVM_LIBC_OPCODE(19),
  LIBC_UNGETC = LLVM_LIBC_OPCODE(20),
  LIBC_PRINTF_TO_STDOUT = LLVM_LIBC_OPCODE(21),
  LIBC_PRINTF_TO_STDERR = LLVM_LIBC_OPCODE(22),
  LIBC_PRINTF_TO_STREAM = LLVM_LIBC_OPCODE(23),
  LIBC_PRINTF_TO_STDOUT_PACKED = LLVM_LIBC_OPCODE(24),
  LIBC_PRINTF_TO_STDERR_PACKED = LLVM_LIBC_OPCODE(25),
  LIBC_PRINTF_TO_STREAM_PACKED = LLVM_LIBC_OPCODE(26),
  LIBC_REMOVE = LLVM_LIBC_OPCODE(27),
  LIBC_RENAME = LLVM_LIBC_OPCODE(28),
  LIBC_SYSTEM = LLVM_LIBC_OPCODE(29),
  LIBC_LAST = 0xFFFFFFFF,
} rpc_opcode_t;

#undef LLVM_LIBC_OPCODE

namespace rpc {
// The implementation of this function currently lives in the utility directory
// at 'utils/gpu/server/rpc_server.cpp'.
rpc::Status handle_libc_opcodes(rpc::Server::Port &port, uint32_t num_lanes);
} // namespace rpc

#endif // LLVM_LIBC_SHARED_RPC_OPCODES_H
