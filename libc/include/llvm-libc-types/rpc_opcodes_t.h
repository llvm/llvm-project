//===-- Definition of RPC opcodes -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_RPC_OPCODE_H__
#define __LLVM_LIBC_TYPES_RPC_OPCODE_H__

typedef enum {
  RPC_NOOP = 0,
  RPC_EXIT,
  RPC_WRITE_TO_STDOUT,
  RPC_WRITE_TO_STDERR,
  RPC_WRITE_TO_STREAM,
  RPC_WRITE_TO_STDOUT_NEWLINE,
  RPC_READ_FROM_STREAM,
  RPC_OPEN_FILE,
  RPC_CLOSE_FILE,
  RPC_MALLOC,
  RPC_FREE,
  RPC_HOST_CALL,
  RPC_ABORT,
  RPC_FEOF,
  RPC_FERROR,
  RPC_CLEARERR,
  RPC_FSEEK,
  RPC_FTELL,
  RPC_FFLUSH,
  RPC_LAST = 0xFFFF,
} rpc_opcode_t;

#endif // __LLVM_LIBC_TYPES_RPC_OPCODE_H__
