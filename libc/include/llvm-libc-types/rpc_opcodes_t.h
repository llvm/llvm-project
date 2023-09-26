//===-- Definition of RPC opcodes -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_RPC_OPCODE_H__
#define __LLVM_LIBC_TYPES_RPC_OPCODE_H__

typedef enum : unsigned short {
  RPC_NOOP = 0,
  RPC_EXIT = 1,
  RPC_WRITE_TO_STDOUT = 2,
  RPC_WRITE_TO_STDERR = 3,
  RPC_WRITE_TO_STREAM = 4,
  RPC_WRITE_TO_STDOUT_NEWLINE = 5,
  RPC_READ_FROM_STREAM = 6,
  RPC_OPEN_FILE = 7,
  RPC_CLOSE_FILE = 8,
  RPC_MALLOC = 9,
  RPC_FREE = 10,
  RPC_HOST_CALL = 11,
  RPC_ABORT = 12,
  RPC_FEOF = 13,
  RPC_FERROR = 14,
  RPC_CLEARERR = 15,
  RPC_FSEEK = 16,
  RPC_FTELL = 17,
  RPC_FFLUSH = 18,
} rpc_opcode_t;

#endif // __LLVM_LIBC_TYPES_RPC_OPCODE_H__
