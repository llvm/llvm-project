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
  RPC_OPEN_FILE = 5,
  RPC_CLOSE_FILE = 6,
  RPC_MALLOC = 7,
  RPC_FREE = 8,
  // TODO: Move these out of here and handle then with custom handlers in the
  // loader.
  RPC_TEST_INCREMENT = 1000,
  RPC_TEST_INTERFACE = 1001,
  RPC_TEST_STREAM = 1002,
} rpc_opcode_t;

#endif // __LLVM_LIBC_TYPES_RPC_OPCODE_H__
