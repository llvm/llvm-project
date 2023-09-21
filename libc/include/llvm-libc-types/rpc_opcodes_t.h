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
  RPC_READ_FROM_STREAM = 5,
  RPC_OPEN_FILE = 6,
  RPC_CLOSE_FILE = 7,
  RPC_MALLOC = 8,
  RPC_FREE = 9,
  RPC_HOST_CALL = 10,
  RPC_ABORT = 11,
  RPC_FEOF = 12,
  RPC_FERROR = 13,
  RPC_CLEARERR = 14,
} rpc_opcode_t;

#endif // __LLVM_LIBC_TYPES_RPC_OPCODE_H__
