//===-- runtime/descriptor-io.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "descriptor-io.h"

namespace Fortran::runtime::io::descr {

// User-defined derived type formatted I/O (maybe)
std::optional<bool> DefinedFormattedIo(IoStatementState &io,
    const Descriptor &descriptor, const typeInfo::SpecialBinding &special) {
  std::optional<DataEdit> peek{io.GetNextDataEdit(0 /*to peek at it*/)};
  if (peek &&
      (peek->descriptor == DataEdit::DefinedDerivedType ||
          peek->descriptor == DataEdit::ListDirected)) {
    // User-defined derived type formatting
    IoErrorHandler &handler{io.GetIoErrorHandler()};
    DataEdit edit{*io.GetNextDataEdit()}; // consume it this time
    RUNTIME_CHECK(handler, edit.descriptor == peek->descriptor);
    char ioType[2 + edit.maxIoTypeChars];
    auto ioTypeLen{std::size_t{2} /*"DT"*/ + edit.ioTypeChars};
    if (edit.descriptor == DataEdit::DefinedDerivedType) {
      ioType[0] = 'D';
      ioType[1] = 'T';
      std::memcpy(ioType + 2, edit.ioType, edit.ioTypeChars);
    } else {
      std::strcpy(
          ioType, io.mutableModes().inNamelist ? "NAMELIST" : "LISTDIRECTED");
      ioTypeLen = std::strlen(ioType);
    }
    StaticDescriptor<1, true> statDesc;
    Descriptor &vListDesc{statDesc.descriptor()};
    vListDesc.Establish(TypeCategory::Integer, sizeof(int), nullptr, 1);
    vListDesc.set_base_addr(edit.vList);
    vListDesc.GetDimension(0).SetBounds(1, edit.vListEntries);
    vListDesc.GetDimension(0).SetByteStride(
        static_cast<SubscriptValue>(sizeof(int)));
    ExternalFileUnit *actualExternal{io.GetExternalFileUnit()};
    ExternalFileUnit *external{actualExternal};
    if (!external) {
      // Create a new unit to service defined I/O for an
      // internal I/O parent.
      external = &ExternalFileUnit::NewUnit(handler, true);
    }
    ChildIo &child{external->PushChildIo(io)};
    int unit{external->unitNumber()};
    int ioStat{IostatOk};
    char ioMsg[100];
    if (special.IsArgDescriptor(0)) {
      auto *p{special.GetProc<void (*)(const Descriptor &, int &, char *,
          const Descriptor &, int &, char *, std::size_t, std::size_t)>()};
      p(descriptor, unit, ioType, vListDesc, ioStat, ioMsg, ioTypeLen,
          sizeof ioMsg);
    } else {
      auto *p{special.GetProc<void (*)(const void *, int &, char *,
          const Descriptor &, int &, char *, std::size_t, std::size_t)>()};
      p(descriptor.raw().base_addr, unit, ioType, vListDesc, ioStat, ioMsg,
          ioTypeLen, sizeof ioMsg);
    }
    handler.Forward(ioStat, ioMsg, sizeof ioMsg);
    external->PopChildIo(child);
    if (!actualExternal) {
      // Close unit created for internal I/O above.
      auto *closing{external->LookUpForClose(external->unitNumber())};
      RUNTIME_CHECK(handler, external == closing);
      external->DestroyClosed();
    }
    return handler.GetIoStat() == IostatOk;
  } else {
    // There's a user-defined I/O subroutine, but there's a FORMAT present and
    // it does not have a DT data edit descriptor, so apply default formatting
    // to the components of the derived type as usual.
    return std::nullopt;
  }
}

// User-defined derived type unformatted I/O
bool DefinedUnformattedIo(IoStatementState &io, const Descriptor &descriptor,
    const typeInfo::SpecialBinding &special) {
  // Unformatted I/O must have an external unit (or child thereof).
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  ExternalFileUnit *external{io.GetExternalFileUnit()};
  RUNTIME_CHECK(handler, external != nullptr);
  ChildIo &child{external->PushChildIo(io)};
  int unit{external->unitNumber()};
  int ioStat{IostatOk};
  char ioMsg[100];
  if (special.IsArgDescriptor(0)) {
    auto *p{special.GetProc<void (*)(
        const Descriptor &, int &, int &, char *, std::size_t)>()};
    p(descriptor, unit, ioStat, ioMsg, sizeof ioMsg);
  } else {
    auto *p{special.GetProc<void (*)(
        const void *, int &, int &, char *, std::size_t)>()};
    p(descriptor.raw().base_addr, unit, ioStat, ioMsg, sizeof ioMsg);
  }
  handler.Forward(ioStat, ioMsg, sizeof ioMsg);
  external->PopChildIo(child);
  return handler.GetIoStat() == IostatOk;
}

} // namespace Fortran::runtime::io::descr
