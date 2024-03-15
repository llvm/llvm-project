//===-- runtime/descriptor-io.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "descriptor-io.h"
#include "flang/Common/restorer.h"

namespace Fortran::runtime::io::descr {

// Defined formatted I/O (maybe)
Fortran::common::optional<bool> DefinedFormattedIo(IoStatementState &io,
    const Descriptor &descriptor, const typeInfo::DerivedType &derived,
    const typeInfo::SpecialBinding &special,
    const SubscriptValue subscripts[]) {
  Fortran::common::optional<DataEdit> peek{
      io.GetNextDataEdit(0 /*to peek at it*/)};
  if (peek &&
      (peek->descriptor == DataEdit::DefinedDerivedType ||
          peek->descriptor == DataEdit::ListDirected)) {
    // Defined formatting
    IoErrorHandler &handler{io.GetIoErrorHandler()};
    DataEdit edit{*io.GetNextDataEdit(1)}; // now consume it; no repeats
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
    StaticDescriptor<1, true> vListStatDesc;
    Descriptor &vListDesc{vListStatDesc.descriptor()};
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
    // Child formatted I/O is nonadvancing by definition (F'2018 12.6.2.4).
    auto restorer{common::ScopedSet(io.mutableModes().nonAdvancing, true)};
    int unit{external->unitNumber()};
    int ioStat{IostatOk};
    char ioMsg[100];
    Fortran::common::optional<std::int64_t> startPos;
    if (edit.descriptor == DataEdit::DefinedDerivedType &&
        special.which() == typeInfo::SpecialBinding::Which::ReadFormatted) {
      // DT is an edit descriptor so everything that the child
      // I/O subroutine reads counts towards READ(SIZE=).
      startPos = io.InquirePos();
    }
    if (special.IsArgDescriptor(0)) {
      // "dtv" argument is "class(t)", pass a descriptor
      auto *p{special.GetProc<void (*)(const Descriptor &, int &, char *,
          const Descriptor &, int &, char *, std::size_t, std::size_t)>()};
      StaticDescriptor<1, true, 10 /*?*/> elementStatDesc;
      Descriptor &elementDesc{elementStatDesc.descriptor()};
      elementDesc.Establish(
          derived, nullptr, 0, nullptr, CFI_attribute_pointer);
      elementDesc.set_base_addr(descriptor.Element<char>(subscripts));
      p(elementDesc, unit, ioType, vListDesc, ioStat, ioMsg, ioTypeLen,
          sizeof ioMsg);
    } else {
      // "dtv" argument is "type(t)", pass a raw pointer
      auto *p{special.GetProc<void (*)(const void *, int &, char *,
          const Descriptor &, int &, char *, std::size_t, std::size_t)>()};
      p(descriptor.Element<char>(subscripts), unit, ioType, vListDesc, ioStat,
          ioMsg, ioTypeLen, sizeof ioMsg);
    }
    handler.Forward(ioStat, ioMsg, sizeof ioMsg);
    external->PopChildIo(child);
    if (!actualExternal) {
      // Close unit created for internal I/O above.
      auto *closing{external->LookUpForClose(external->unitNumber())};
      RUNTIME_CHECK(handler, external == closing);
      external->DestroyClosed();
    }
    if (startPos) {
      io.GotChar(io.InquirePos() - *startPos);
    }
    return handler.GetIoStat() == IostatOk;
  } else {
    // There's a defined I/O subroutine, but there's a FORMAT present and
    // it does not have a DT data edit descriptor, so apply default formatting
    // to the components of the derived type as usual.
    return Fortran::common::nullopt;
  }
}

// Defined unformatted I/O
bool DefinedUnformattedIo(IoStatementState &io, const Descriptor &descriptor,
    const typeInfo::DerivedType &derived,
    const typeInfo::SpecialBinding &special) {
  // Unformatted I/O must have an external unit (or child thereof).
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  ExternalFileUnit *external{io.GetExternalFileUnit()};
  if (!external) { // INQUIRE(IOLENGTH=)
    handler.SignalError(IostatNonExternalDefinedUnformattedIo);
    return false;
  }
  ChildIo &child{external->PushChildIo(io)};
  int unit{external->unitNumber()};
  int ioStat{IostatOk};
  char ioMsg[100];
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  if (special.IsArgDescriptor(0)) {
    // "dtv" argument is "class(t)", pass a descriptor
    auto *p{special.GetProc<void (*)(
        const Descriptor &, int &, int &, char *, std::size_t)>()};
    StaticDescriptor<1, true, 10 /*?*/> elementStatDesc;
    Descriptor &elementDesc{elementStatDesc.descriptor()};
    elementDesc.Establish(derived, nullptr, 0, nullptr, CFI_attribute_pointer);
    for (; numElements-- > 0; descriptor.IncrementSubscripts(subscripts)) {
      elementDesc.set_base_addr(descriptor.Element<char>(subscripts));
      p(elementDesc, unit, ioStat, ioMsg, sizeof ioMsg);
      if (ioStat != IostatOk) {
        break;
      }
    }
  } else {
    // "dtv" argument is "type(t)", pass a raw pointer
    auto *p{special.GetProc<void (*)(
        const void *, int &, int &, char *, std::size_t)>()};
    for (; numElements-- > 0; descriptor.IncrementSubscripts(subscripts)) {
      p(descriptor.Element<char>(subscripts), unit, ioStat, ioMsg,
          sizeof ioMsg);
      if (ioStat != IostatOk) {
        break;
      }
    }
  }
  handler.Forward(ioStat, ioMsg, sizeof ioMsg);
  external->PopChildIo(child);
  return handler.GetIoStat() == IostatOk;
}

} // namespace Fortran::runtime::io::descr
