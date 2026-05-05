//===-- lib/runtime/io-api-gpu.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the subset of the I/O statement API needed for basic list-directed
// output (PRINT *) of intrinsic types for the GPU.
//
// The RPC interface forwards each runtime call from the client to the server
// using a shared buffer. These calls are buffered on the server, so only the
// return values from 'Begin' and 'EndIoStatement' are meaningful.

#include "io-api-gpu.h"
#include "flang/Runtime/io-api.h"

#include <shared/rpc.h>
#include <shared/rpc_dispatch.h>

namespace Fortran::runtime::io {
// A weak reference to the RPC client used to submit calls to the server.
[[gnu::weak, gnu::visibility("protected")]] rpc::Client client asm(
    "__llvm_rpc_client");

RT_EXT_API_GROUP_BEGIN

Cookie IODEF(BeginExternalListOutput)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return rpc::dispatch<BeginExternalListOutput_Opcode,
      IONAME(BeginExternalListOutput)>(
      client, unitNumber, sourceFile, sourceLine);
}

Cookie IODEF(BeginExternalFormattedOutput)(const char *format,
    std::size_t formatLength, const Descriptor *formatDescriptor,
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return rpc::dispatch<BeginExternalFormattedOutput_Opcode,
      IONAME(BeginExternalFormattedOutput)>(client,
      rpc::span<const char>{format, formatLength}, formatLength,
      formatDescriptor, unitNumber, sourceFile, sourceLine);
}

void IODEF(EnableHandlers)(Cookie cookie, bool hasIoStat, bool hasErr,
    bool hasEnd, bool hasEor, bool hasIoMsg) {
  return rpc::dispatch<EnableHandlers_Opcode, IONAME(EnableHandlers)>(
      client, cookie, hasIoStat, hasErr, hasEnd, hasEor, hasIoMsg);
}

enum Iostat IODEF(EndIoStatement)(Cookie cookie) {
  return rpc::dispatch<EndIoStatement_Opcode, IONAME(EndIoStatement)>(
      client, cookie);
}

bool IODEF(OutputInteger8)(Cookie cookie, std::int8_t n) {
  return rpc::dispatch<OutputInteger8_Opcode, IONAME(OutputInteger8)>(
      client, cookie, n);
}

bool IODEF(OutputInteger16)(Cookie cookie, std::int16_t n) {
  return rpc::dispatch<OutputInteger16_Opcode, IONAME(OutputInteger16)>(
      client, cookie, n);
}

bool IODEF(OutputInteger32)(Cookie cookie, std::int32_t n) {
  return rpc::dispatch<OutputInteger32_Opcode, IONAME(OutputInteger32)>(
      client, cookie, n);
}

bool IODEF(OutputInteger64)(Cookie cookie, std::int64_t n) {
  return rpc::dispatch<OutputInteger64_Opcode, IONAME(OutputInteger64)>(
      client, cookie, n);
}

#ifdef __SIZEOF_INT128__
bool IODEF(OutputInteger128)(Cookie cookie, common::int128_t n) {
  return rpc::dispatch<OutputInteger128_Opcode, IONAME(OutputInteger128)>(
      client, cookie, n);
}
#endif

bool IODEF(OutputReal32)(Cookie cookie, float x) {
  return rpc::dispatch<OutputReal32_Opcode, IONAME(OutputReal32)>(
      client, cookie, x);
}

bool IODEF(OutputReal64)(Cookie cookie, double x) {
  return rpc::dispatch<OutputReal64_Opcode, IONAME(OutputReal64)>(
      client, cookie, x);
}

bool IODEF(OutputComplex32)(Cookie cookie, float re, float im) {
  return rpc::dispatch<OutputComplex32_Opcode, IONAME(OutputComplex32)>(
      client, cookie, re, im);
}

bool IODEF(OutputComplex64)(Cookie cookie, double re, double im) {
  return rpc::dispatch<OutputComplex64_Opcode, IONAME(OutputComplex64)>(
      client, cookie, re, im);
}

bool IODEF(OutputAscii)(Cookie cookie, const char *x, std::size_t length) {
  return rpc::dispatch<OutputAscii_Opcode, IONAME(OutputAscii)>(
      client, cookie, rpc::span<const char>{x, length}, length);
}

bool IODEF(OutputCharacter)(
    Cookie cookie, const char *x, std::size_t length, int kind) {
  return rpc::dispatch<OutputCharacter_Opcode, IONAME(OutputCharacter)>(
      client, cookie, rpc::span<const char>{x, length * kind}, length, kind);
}

bool IODEF(OutputLogical)(Cookie cookie, bool truth) {
  return rpc::dispatch<OutputLogical_Opcode, IONAME(OutputLogical)>(
      client, cookie, truth);
}

RT_EXT_API_GROUP_END
} // namespace Fortran::runtime::io
