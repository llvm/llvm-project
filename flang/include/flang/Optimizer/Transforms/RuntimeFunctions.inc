//===-- Optimizer/Transforms/RuntimeFunctions.inc ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KNOWN_IO_FUNC
#error "Define KNOWN_IO_FUNC before including this file"
#endif
#ifndef KNOWN_RUNTIME_FUNC
#error "Define KNOWN_RUNTIME_FUNC before including this file"
#endif

// Fortran runtime functions that SetRuntimeCallAttributesPass recognizes.
// WARNING: if you add a function entry here, you must make sure
// that the attribute computation callbacks that end up being
// used are correct for this function. If needed, add
// specializations for the types that provide attribute
// computation callbacks in SetRuntimeCallAttributesPass.

// clang-format off
KNOWN_IO_FUNC(BeginBackspace),
KNOWN_IO_FUNC(BeginClose),
KNOWN_IO_FUNC(BeginEndfile),
KNOWN_IO_FUNC(BeginExternalFormattedInput),
KNOWN_IO_FUNC(BeginExternalFormattedOutput),
KNOWN_IO_FUNC(BeginExternalListInput),
KNOWN_IO_FUNC(BeginExternalListOutput),
KNOWN_IO_FUNC(BeginFlush),
KNOWN_IO_FUNC(BeginInquireFile),
KNOWN_IO_FUNC(BeginInquireIoLength),
KNOWN_IO_FUNC(BeginInquireUnit),
KNOWN_IO_FUNC(BeginInternalArrayFormattedInput),
KNOWN_IO_FUNC(BeginInternalArrayFormattedOutput),
KNOWN_IO_FUNC(BeginInternalArrayListInput),
KNOWN_IO_FUNC(BeginInternalArrayListOutput),
KNOWN_IO_FUNC(BeginInternalFormattedInput),
KNOWN_IO_FUNC(BeginInternalFormattedOutput),
KNOWN_IO_FUNC(BeginInternalListInput),
KNOWN_IO_FUNC(BeginInternalListOutput),
KNOWN_IO_FUNC(BeginOpenNewUnit),
KNOWN_IO_FUNC(BeginOpenUnit),
KNOWN_IO_FUNC(BeginRewind),
KNOWN_IO_FUNC(BeginUnformattedInput),
KNOWN_IO_FUNC(BeginUnformattedOutput),
KNOWN_IO_FUNC(BeginWait),
KNOWN_IO_FUNC(BeginWaitAll),
KNOWN_IO_FUNC(CheckUnitNumberInRange128),
KNOWN_IO_FUNC(CheckUnitNumberInRange64),
KNOWN_IO_FUNC(EnableHandlers),
KNOWN_IO_FUNC(EndIoStatement),
KNOWN_IO_FUNC(GetAsynchronousId),
KNOWN_IO_FUNC(GetIoLength),
KNOWN_IO_FUNC(GetIoMsg),
KNOWN_IO_FUNC(GetNewUnit),
KNOWN_IO_FUNC(GetSize),
KNOWN_IO_FUNC(InputAscii),
KNOWN_IO_FUNC(InputComplex32),
KNOWN_IO_FUNC(InputComplex64),
KNOWN_IO_FUNC(InputDerivedType),
KNOWN_IO_FUNC(InputDescriptor),
KNOWN_IO_FUNC(InputInteger),
KNOWN_IO_FUNC(InputLogical),
KNOWN_IO_FUNC(InputNamelist),
KNOWN_IO_FUNC(InputReal32),
KNOWN_IO_FUNC(InputReal64),
KNOWN_IO_FUNC(InquireCharacter),
KNOWN_IO_FUNC(InquireInteger64),
KNOWN_IO_FUNC(InquireLogical),
KNOWN_IO_FUNC(InquirePendingId),
KNOWN_IO_FUNC(OutputAscii),
KNOWN_IO_FUNC(OutputComplex32),
KNOWN_IO_FUNC(OutputComplex64),
KNOWN_IO_FUNC(OutputDerivedType),
KNOWN_IO_FUNC(OutputDescriptor),
KNOWN_IO_FUNC(OutputInteger128),
KNOWN_IO_FUNC(OutputInteger16),
KNOWN_IO_FUNC(OutputInteger32),
KNOWN_IO_FUNC(OutputInteger64),
KNOWN_IO_FUNC(OutputInteger8),
KNOWN_IO_FUNC(OutputLogical),
KNOWN_IO_FUNC(OutputNamelist),
KNOWN_IO_FUNC(OutputReal32),
KNOWN_IO_FUNC(OutputReal64),
KNOWN_IO_FUNC(SetAccess),
KNOWN_IO_FUNC(SetAction),
KNOWN_IO_FUNC(SetAdvance),
KNOWN_IO_FUNC(SetAsynchronous),
KNOWN_IO_FUNC(SetBlank),
KNOWN_IO_FUNC(SetCarriagecontrol),
KNOWN_IO_FUNC(SetConvert),
KNOWN_IO_FUNC(SetDecimal),
KNOWN_IO_FUNC(SetDelim),
KNOWN_IO_FUNC(SetEncoding),
KNOWN_IO_FUNC(SetFile),
KNOWN_IO_FUNC(SetForm),
KNOWN_IO_FUNC(SetPad),
KNOWN_IO_FUNC(SetPos),
KNOWN_IO_FUNC(SetPosition),
KNOWN_IO_FUNC(SetRec),
KNOWN_IO_FUNC(SetRecl),
KNOWN_IO_FUNC(SetRound),
KNOWN_IO_FUNC(SetSign),
KNOWN_IO_FUNC(SetStatus)

// clang-format on

#undef KNOWN_IO_FUNC
#undef KNOWN_RUNTIME_FUNC
