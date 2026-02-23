//===-- lib/runtime/io-api-gpu.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_IO_API_GPU_H_
#define FLANG_RT_RUNTIME_IO_API_GPU_H_

#include <cstdint>

namespace Fortran::runtime::io {
// We reserve the RPC opcodes with 'f' in the MSB for Fortran usage.
constexpr std::uint32_t MakeOpcode(std::uint32_t base) {
  return ('f' << 24) | base;
}

// Opcodes shared between the client and server for each function we support.
enum RPCOpcodes : std::uint32_t {
  BeginExternalListOutput_Opcode = MakeOpcode(0),
  BeginExternalFormattedOutput_Opcode = MakeOpcode(1),
  EnableHandlers_Opcode = MakeOpcode(2),
  EndIoStatement_Opcode = MakeOpcode(3),
  OutputInteger8_Opcode = MakeOpcode(4),
  OutputInteger16_Opcode = MakeOpcode(5),
  OutputInteger32_Opcode = MakeOpcode(6),
  OutputInteger64_Opcode = MakeOpcode(7),
  OutputInteger128_Opcode = MakeOpcode(8),
  OutputReal32_Opcode = MakeOpcode(9),
  OutputReal64_Opcode = MakeOpcode(10),
  OutputComplex32_Opcode = MakeOpcode(11),
  OutputComplex64_Opcode = MakeOpcode(12),
  OutputAscii_Opcode = MakeOpcode(13),
  OutputCharacter_Opcode = MakeOpcode(14),
  OutputLogical_Opcode = MakeOpcode(15),
};

} // namespace Fortran::runtime::io

#endif // FLANG_RT_RUNTIME_IO_API_GPU_H_
