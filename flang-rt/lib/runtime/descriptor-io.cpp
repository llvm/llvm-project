//===-- lib/runtime/descriptor-io.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "descriptor-io.h"
#include "edit-input.h"
#include "edit-output.h"
#include "unit.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/io-stmt.h"
#include "flang-rt/runtime/namelist.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/type-info.h"
#include "flang-rt/runtime/work-queue.h"
#include "flang/Common/optional.h"
#include "flang/Common/restorer.h"
#include "flang/Common/uint128.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/freestanding-tools.h"

// Implementation of I/O data list item transfers based on descriptors.
// (All I/O items come through here so that the code is exercised for test;
// some scalar I/O data transfer APIs could be changed to bypass their use
// of descriptors in the future for better efficiency.)

namespace Fortran::runtime::io::descr {
RT_OFFLOAD_API_GROUP_BEGIN

template <typename A>
inline RT_API_ATTRS A &ExtractElement(IoStatementState &io,
    const Descriptor &descriptor, const SubscriptValue subscripts[]) {
  A *p{descriptor.Element<A>(subscripts)};
  if (!p) {
    io.GetIoErrorHandler().Crash("Bad address for I/O item -- null base "
                                 "address or subscripts out of range");
  }
  return *p;
}

// Defined formatted I/O (maybe)
static RT_API_ATTRS Fortran::common::optional<bool> DefinedFormattedIo(
    IoStatementState &io, const Descriptor &descriptor,
    const typeInfo::DerivedType &derived,
    const typeInfo::SpecialBinding &special,
    const SubscriptValue subscripts[]) {
  // Look at the next data edit descriptor.  If this is list-directed I/O, the
  // "maxRepeat=0" argument will prevent the input from advancing over an
  // initial '(' that shouldn't be consumed now as the start of a real part.
  Fortran::common::optional<DataEdit> peek{io.GetNextDataEdit(/*maxRepeat=*/0)};
  if (peek &&
      (peek->descriptor == DataEdit::DefinedDerivedType ||
          peek->descriptor == DataEdit::ListDirected ||
          peek->descriptor == DataEdit::ListDirectedRealPart)) {
    // Defined formatting
    IoErrorHandler &handler{io.GetIoErrorHandler()};
    DataEdit edit{peek->descriptor == DataEdit::ListDirectedRealPart
            ? *peek
            : *io.GetNextDataEdit(1)};
    char ioType[2 + edit.maxIoTypeChars];
    auto ioTypeLen{std::size_t{2} /*"DT"*/ + edit.ioTypeChars};
    if (edit.descriptor == DataEdit::DefinedDerivedType) {
      ioType[0] = 'D';
      ioType[1] = 'T';
      std::memcpy(ioType + 2, edit.ioType, edit.ioTypeChars);
    } else {
      runtime::strcpy(
          ioType, io.mutableModes().inNamelist ? "NAMELIST" : "LISTDIRECTED");
      ioTypeLen = runtime::strlen(ioType);
    }
    // V_LIST= argument
    StaticDescriptor<1, true> vListStatDesc;
    Descriptor &vListDesc{vListStatDesc.descriptor()};
    bool integer8{special.specialCaseFlag()};
    std::int64_t vList64[edit.maxVListEntries];
    if (integer8) {
      // Convert v_list values to INTEGER(8)
      for (int j{0}; j < edit.vListEntries; ++j) {
        vList64[j] = edit.vList[j];
      }
      vListDesc.Establish(
          TypeCategory::Integer, sizeof(std::int64_t), nullptr, 1);
      vListDesc.set_base_addr(vList64);
      vListDesc.GetDimension(0).SetBounds(1, edit.vListEntries);
      vListDesc.GetDimension(0).SetByteStride(
          static_cast<SubscriptValue>(sizeof(std::int64_t)));
    } else {
      vListDesc.Establish(TypeCategory::Integer, sizeof(int), nullptr, 1);
      vListDesc.set_base_addr(edit.vList);
      vListDesc.GetDimension(0).SetBounds(1, edit.vListEntries);
      vListDesc.GetDimension(0).SetByteStride(
          static_cast<SubscriptValue>(sizeof(int)));
    }
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
    std::int32_t unit{external->unitNumber()};
    std::int32_t ioStat{IostatOk};
    char ioMsg[100];
    Fortran::common::optional<std::int64_t> startPos;
    if (edit.descriptor == DataEdit::DefinedDerivedType &&
        special.which() == typeInfo::SpecialBinding::Which::ReadFormatted) {
      // DT is an edit descriptor, so everything that the child
      // I/O subroutine reads counts towards READ(SIZE=).
      startPos = io.InquirePos();
    }
    const auto *bindings{
        derived.binding().OffsetElement<const typeInfo::Binding>()};
    if (special.IsArgDescriptor(0)) {
      // "dtv" argument is "class(t)", pass a descriptor
      StaticDescriptor<1, true, 10 /*?*/> elementStatDesc;
      Descriptor &elementDesc{elementStatDesc.descriptor()};
      elementDesc.Establish(
          derived, nullptr, 0, nullptr, CFI_attribute_pointer);
      elementDesc.set_base_addr(descriptor.Element<char>(subscripts));
      if (integer8) { // 64-bit UNIT=/IOSTAT=
        std::int64_t unit64{unit};
        std::int64_t ioStat64{ioStat};
        auto *p{special.GetProc<void (*)(const Descriptor &, std::int64_t &,
            char *, const Descriptor &, std::int64_t &, char *, std::size_t,
            std::size_t)>(bindings)};
        p(elementDesc, unit64, ioType, vListDesc, ioStat64, ioMsg, ioTypeLen,
            sizeof ioMsg);
        ioStat = ioStat64;
      } else { // 32-bit UNIT=/IOSTAT=
        auto *p{special.GetProc<void (*)(const Descriptor &, std::int32_t &,
            char *, const Descriptor &, std::int32_t &, char *, std::size_t,
            std::size_t)>(bindings)};
        p(elementDesc, unit, ioType, vListDesc, ioStat, ioMsg, ioTypeLen,
            sizeof ioMsg);
      }
    } else {
      // "dtv" argument is "type(t)", pass a raw pointer
      if (integer8) { // 64-bit UNIT= and IOSTAT=
        std::int64_t unit64{unit};
        std::int64_t ioStat64{ioStat};
        auto *p{special.GetProc<void (*)(const void *, std::int64_t &, char *,
            const Descriptor &, std::int64_t &, char *, std::size_t,
            std::size_t)>(bindings)};
        p(descriptor.Element<char>(subscripts), unit64, ioType, vListDesc,
            ioStat64, ioMsg, ioTypeLen, sizeof ioMsg);
        ioStat = ioStat64;
      } else { // 32-bit UNIT= and IOSTAT=
        auto *p{special.GetProc<void (*)(const void *, std::int32_t &, char *,
            const Descriptor &, std::int32_t &, char *, std::size_t,
            std::size_t)>(bindings)};
        p(descriptor.Element<char>(subscripts), unit, ioType, vListDesc, ioStat,
            ioMsg, ioTypeLen, sizeof ioMsg);
      }
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
  } else if (peek && peek->descriptor == DataEdit::ListDirectedNullValue) {
    return false;
  } else {
    // There's a defined I/O subroutine, but there's a FORMAT present and
    // it does not have a DT data edit descriptor, so apply default formatting
    // to the components of the derived type as usual.
    return Fortran::common::nullopt;
  }
}

// Defined unformatted I/O
static RT_API_ATTRS bool DefinedUnformattedIo(IoStatementState &io,
    const Descriptor &descriptor, const typeInfo::DerivedType &derived,
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
  const auto *bindings{
      derived.binding().OffsetElement<const typeInfo::Binding>()};
  if (special.IsArgDescriptor(0)) {
    // "dtv" argument is "class(t)", pass a descriptor
    auto *p{special.GetProc<void (*)(
        const Descriptor &, int &, int &, char *, std::size_t)>(bindings)};
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
    auto *p{special
            .GetProc<void (*)(const void *, int &, int &, char *, std::size_t)>(
                bindings)};
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

// Per-category descriptor-based I/O templates

// TODO (perhaps as a nontrivial but small starter project): implement
// automatic repetition counts, like "10*3.14159", for list-directed and
// NAMELIST array output.

template <int KIND, Direction DIR>
inline RT_API_ATTRS bool FormattedIntegerIO(IoStatementState &io,
    const Descriptor &descriptor, [[maybe_unused]] bool isSigned) {
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  using IntType = CppTypeFor<common::TypeCategory::Integer, KIND>;
  bool anyInput{false};
  for (std::size_t j{0}; j < numElements; ++j) {
    if (auto edit{io.GetNextDataEdit()}) {
      IntType &x{ExtractElement<IntType>(io, descriptor, subscripts)};
      if constexpr (DIR == Direction::Output) {
        if (!EditIntegerOutput<KIND>(io, *edit, x, isSigned)) {
          return false;
        }
      } else if (edit->descriptor != DataEdit::ListDirectedNullValue) {
        if (EditIntegerInput(
                io, *edit, reinterpret_cast<void *>(&x), KIND, isSigned)) {
          anyInput = true;
        } else {
          return anyInput && edit->IsNamelist();
        }
      }
      if (!descriptor.IncrementSubscripts(subscripts) && j + 1 < numElements) {
        io.GetIoErrorHandler().Crash(
            "FormattedIntegerIO: subscripts out of bounds");
      }
    } else {
      return false;
    }
  }
  return true;
}

template <int KIND, Direction DIR>
inline RT_API_ATTRS bool FormattedRealIO(
    IoStatementState &io, const Descriptor &descriptor) {
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  using RawType = typename RealOutputEditing<KIND>::BinaryFloatingPoint;
  bool anyInput{false};
  for (std::size_t j{0}; j < numElements; ++j) {
    if (auto edit{io.GetNextDataEdit()}) {
      RawType &x{ExtractElement<RawType>(io, descriptor, subscripts)};
      if constexpr (DIR == Direction::Output) {
        if (!RealOutputEditing<KIND>{io, x}.Edit(*edit)) {
          return false;
        }
      } else if (edit->descriptor != DataEdit::ListDirectedNullValue) {
        if (EditRealInput<KIND>(io, *edit, reinterpret_cast<void *>(&x))) {
          anyInput = true;
        } else {
          return anyInput && edit->IsNamelist();
        }
      }
      if (!descriptor.IncrementSubscripts(subscripts) && j + 1 < numElements) {
        io.GetIoErrorHandler().Crash(
            "FormattedRealIO: subscripts out of bounds");
      }
    } else {
      return false;
    }
  }
  return true;
}

template <int KIND, Direction DIR>
inline RT_API_ATTRS bool FormattedComplexIO(
    IoStatementState &io, const Descriptor &descriptor) {
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  bool isListOutput{
      io.get_if<ListDirectedStatementState<Direction::Output>>() != nullptr};
  using RawType = typename RealOutputEditing<KIND>::BinaryFloatingPoint;
  bool anyInput{false};
  for (std::size_t j{0}; j < numElements; ++j) {
    RawType *x{&ExtractElement<RawType>(io, descriptor, subscripts)};
    if (isListOutput) {
      DataEdit rEdit, iEdit;
      rEdit.descriptor = DataEdit::ListDirectedRealPart;
      iEdit.descriptor = DataEdit::ListDirectedImaginaryPart;
      rEdit.modes = iEdit.modes = io.mutableModes();
      if (!RealOutputEditing<KIND>{io, x[0]}.Edit(rEdit) ||
          !RealOutputEditing<KIND>{io, x[1]}.Edit(iEdit)) {
        return false;
      }
    } else {
      for (int k{0}; k < 2; ++k, ++x) {
        auto edit{io.GetNextDataEdit()};
        if (!edit) {
          return false;
        } else if constexpr (DIR == Direction::Output) {
          if (!RealOutputEditing<KIND>{io, *x}.Edit(*edit)) {
            return false;
          }
        } else if (edit->descriptor == DataEdit::ListDirectedNullValue) {
          break;
        } else if (EditRealInput<KIND>(
                       io, *edit, reinterpret_cast<void *>(x))) {
          anyInput = true;
        } else {
          return anyInput && edit->IsNamelist();
        }
      }
    }
    if (!descriptor.IncrementSubscripts(subscripts) && j + 1 < numElements) {
      io.GetIoErrorHandler().Crash(
          "FormattedComplexIO: subscripts out of bounds");
    }
  }
  return true;
}

template <typename A, Direction DIR>
inline RT_API_ATTRS bool FormattedCharacterIO(
    IoStatementState &io, const Descriptor &descriptor) {
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  std::size_t length{descriptor.ElementBytes() / sizeof(A)};
  auto *listOutput{io.get_if<ListDirectedStatementState<Direction::Output>>()};
  bool anyInput{false};
  for (std::size_t j{0}; j < numElements; ++j) {
    A *x{&ExtractElement<A>(io, descriptor, subscripts)};
    if (listOutput) {
      if (!ListDirectedCharacterOutput(io, *listOutput, x, length)) {
        return false;
      }
    } else if (auto edit{io.GetNextDataEdit()}) {
      if constexpr (DIR == Direction::Output) {
        if (!EditCharacterOutput(io, *edit, x, length)) {
          return false;
        }
      } else { // input
        if (edit->descriptor != DataEdit::ListDirectedNullValue) {
          if (EditCharacterInput(io, *edit, x, length)) {
            anyInput = true;
          } else {
            return anyInput && edit->IsNamelist();
          }
        }
      }
    } else {
      return false;
    }
    if (!descriptor.IncrementSubscripts(subscripts) && j + 1 < numElements) {
      io.GetIoErrorHandler().Crash(
          "FormattedCharacterIO: subscripts out of bounds");
    }
  }
  return true;
}

template <int KIND, Direction DIR>
inline RT_API_ATTRS bool FormattedLogicalIO(
    IoStatementState &io, const Descriptor &descriptor) {
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  auto *listOutput{io.get_if<ListDirectedStatementState<Direction::Output>>()};
  using IntType = CppTypeFor<TypeCategory::Integer, KIND>;
  bool anyInput{false};
  for (std::size_t j{0}; j < numElements; ++j) {
    IntType &x{ExtractElement<IntType>(io, descriptor, subscripts)};
    if (listOutput) {
      if (!ListDirectedLogicalOutput(io, *listOutput, x != 0)) {
        return false;
      }
    } else if (auto edit{io.GetNextDataEdit()}) {
      if constexpr (DIR == Direction::Output) {
        if (!EditLogicalOutput(io, *edit, x != 0)) {
          return false;
        }
      } else {
        if (edit->descriptor != DataEdit::ListDirectedNullValue) {
          bool truth{};
          if (EditLogicalInput(io, *edit, truth)) {
            x = truth;
            anyInput = true;
          } else {
            return anyInput && edit->IsNamelist();
          }
        }
      }
    } else {
      return false;
    }
    if (!descriptor.IncrementSubscripts(subscripts) && j + 1 < numElements) {
      io.GetIoErrorHandler().Crash(
          "FormattedLogicalIO: subscripts out of bounds");
    }
  }
  return true;
}

template <Direction DIR>
RT_API_ATTRS int DerivedIoTicket<DIR>::Continue(WorkQueue &workQueue) {
  while (!IsComplete()) {
    if (component_->genre() == typeInfo::Component::Genre::Data) {
      // Create a descriptor for the component
      Descriptor &compDesc{componentDescriptor_.descriptor()};
      component_->CreatePointerDescriptor(
          compDesc, instance_, io_.GetIoErrorHandler(), subscripts_);
      Advance();
      if (int status{workQueue.BeginDescriptorIo<DIR>(
              io_, compDesc, table_, anyIoTookPlace_)};
          status != StatOk) {
        return status;
      }
    } else {
      // Component is itself a descriptor
      char *pointer{
          instance_.Element<char>(subscripts_) + component_->offset()};
      const Descriptor &compDesc{
          *reinterpret_cast<const Descriptor *>(pointer)};
      Advance();
      if (compDesc.IsAllocated()) {
        if (int status{workQueue.BeginDescriptorIo<DIR>(
                io_, compDesc, table_, anyIoTookPlace_)};
            status != StatOk) {
          return status;
        }
      }
    }
  }
  return StatOk;
}

template RT_API_ATTRS int DerivedIoTicket<Direction::Output>::Continue(
    WorkQueue &);
template RT_API_ATTRS int DerivedIoTicket<Direction::Input>::Continue(
    WorkQueue &);

template <Direction DIR>
RT_API_ATTRS int DescriptorIoTicket<DIR>::Begin(WorkQueue &workQueue) {
  IoErrorHandler &handler{io_.GetIoErrorHandler()};
  if (handler.InError()) {
    return handler.GetIoStat();
  }
  if (!io_.get_if<IoDirectionState<DIR>>()) {
    handler.Crash("DescriptorIO() called for wrong I/O direction");
    return handler.GetIoStat();
  }
  if constexpr (DIR == Direction::Input) {
    if (!io_.BeginReadingRecord()) {
      return StatOk;
    }
  }
  if (!io_.get_if<FormattedIoStatementState<DIR>>()) {
    // Unformatted I/O
    IoErrorHandler &handler{io_.GetIoErrorHandler()};
    const DescriptorAddendum *addendum{instance_.Addendum()};
    if (const typeInfo::DerivedType *type{
            addendum ? addendum->derivedType() : nullptr}) {
      // derived type unformatted I/O
      if (DIR == Direction::Input || !io_.get_if<InquireIOLengthState>()) {
        if (table_) {
          if (const auto *definedIo{table_->Find(*type,
                  DIR == Direction::Input
                      ? common::DefinedIo::ReadUnformatted
                      : common::DefinedIo::WriteUnformatted)}) {
            if (definedIo->subroutine) {
              std::uint8_t isArgDescriptorSet{0};
              if (definedIo->flags & IsDtvArgPolymorphic) {
                isArgDescriptorSet = 1;
              }
              typeInfo::SpecialBinding special{DIR == Direction::Input
                      ? typeInfo::SpecialBinding::Which::ReadUnformatted
                      : typeInfo::SpecialBinding::Which::WriteUnformatted,
                  definedIo->subroutine, isArgDescriptorSet,
                  /*IsTypeBound=*/false,
                  /*specialCaseFlag=*/!!(definedIo->flags & DefinedIoInteger8)};
              if (DefinedUnformattedIo(io_, instance_, *type, special)) {
                anyIoTookPlace_ = true;
                return StatOk;
              }
            } else {
              int status{workQueue.BeginDerivedIo<DIR>(
                  io_, instance_, *type, table_, anyIoTookPlace_)};
              return status == StatContinue ? StatOk : status; // done here
            }
          }
        }
        if (const typeInfo::SpecialBinding *special{
                type->FindSpecialBinding(DIR == Direction::Input
                        ? typeInfo::SpecialBinding::Which::ReadUnformatted
                        : typeInfo::SpecialBinding::Which::WriteUnformatted)}) {
          if (!table_ || !table_->ignoreNonTbpEntries ||
              special->IsTypeBound()) {
            // defined derived type unformatted I/O
            if (DefinedUnformattedIo(io_, instance_, *type, *special)) {
              anyIoTookPlace_ = true;
              return StatOk;
            } else {
              return IostatEnd;
            }
          }
        }
      }
      // Default derived type unformatted I/O
      // TODO: If no component at any level has defined READ or WRITE
      // (as appropriate), the elements are contiguous, and no byte swapping
      // is active, do a block transfer via the code below.
      int status{workQueue.BeginDerivedIo<DIR>(
          io_, instance_, *type, table_, anyIoTookPlace_)};
      return status == StatContinue ? StatOk : status; // done here
    } else {
      // intrinsic type unformatted I/O
      auto *externalUnf{io_.get_if<ExternalUnformattedIoStatementState<DIR>>()};
      ChildUnformattedIoStatementState<DIR> *childUnf{nullptr};
      InquireIOLengthState *inq{nullptr};
      bool swapEndianness{false};
      if (externalUnf) {
        swapEndianness = externalUnf->unit().swapEndianness();
      } else {
        childUnf = io_.get_if<ChildUnformattedIoStatementState<DIR>>();
        if (!childUnf) {
          inq = DIR == Direction::Output ? io_.get_if<InquireIOLengthState>()
                                         : nullptr;
          RUNTIME_CHECK(handler, inq != nullptr);
        }
      }
      std::size_t elementBytes{instance_.ElementBytes()};
      std::size_t swappingBytes{elementBytes};
      if (auto maybeCatAndKind{instance_.type().GetCategoryAndKind()}) {
        // Byte swapping units can be smaller than elements, namely
        // for COMPLEX and CHARACTER.
        if (maybeCatAndKind->first == TypeCategory::Character) {
          // swap each character position independently
          swappingBytes = maybeCatAndKind->second; // kind
        } else if (maybeCatAndKind->first == TypeCategory::Complex) {
          // swap real and imaginary components independently
          swappingBytes /= 2;
        }
      }
      using CharType =
          std::conditional_t<DIR == Direction::Output, const char, char>;
      auto Transfer{[=](CharType &x, std::size_t totalBytes) -> bool {
        if constexpr (DIR == Direction::Output) {
          return externalUnf ? externalUnf->Emit(&x, totalBytes, swappingBytes)
              : childUnf     ? childUnf->Emit(&x, totalBytes, swappingBytes)
                             : inq->Emit(&x, totalBytes, swappingBytes);
        } else {
          return externalUnf
              ? externalUnf->Receive(&x, totalBytes, swappingBytes)
              : childUnf->Receive(&x, totalBytes, swappingBytes);
        }
      }};
      if (!swapEndianness &&
          instance_.IsContiguous()) { // contiguous unformatted I/O
        char &x{ExtractElement<char>(io_, instance_, subscripts_)};
        if (Transfer(x, elements_ * elementBytes)) {
          anyIoTookPlace_ = true;
        } else {
          return IostatEnd;
        }
      } else { // non-contiguous or byte-swapped intrinsic type unformatted I/O
        for (; !IsComplete(); Advance()) {
          char &x{ExtractElement<char>(io_, instance_, subscripts_)};
          if (Transfer(x, elementBytes)) {
            anyIoTookPlace_ = true;
          } else {
            return IostatEnd;
          }
        }
      }
    }
    // Unformatted I/O never needs to call Continue().
    return StatOk;
  }
  // Formatted I/O
  if (auto catAndKind{instance_.type().GetCategoryAndKind()}) {
    TypeCategory cat{catAndKind->first};
    int kind{catAndKind->second};
    bool any{false};
    switch (cat) {
    case TypeCategory::Integer:
      switch (kind) {
      case 1:
        any = FormattedIntegerIO<1, DIR>(io_, instance_, true);
        break;
      case 2:
        any = FormattedIntegerIO<2, DIR>(io_, instance_, true);
        break;
      case 4:
        any = FormattedIntegerIO<4, DIR>(io_, instance_, true);
        break;
      case 8:
        any = FormattedIntegerIO<8, DIR>(io_, instance_, true);
        break;
      case 16:
        any = FormattedIntegerIO<16, DIR>(io_, instance_, true);
        break;
      default:
        handler.Crash(
            "not yet implemented: INTEGER(KIND=%d) in formatted IO", kind);
        return IostatEnd;
      }
      break;
    case TypeCategory::Unsigned:
      switch (kind) {
      case 1:
        any = FormattedIntegerIO<1, DIR>(io_, instance_, false);
        break;
      case 2:
        any = FormattedIntegerIO<2, DIR>(io_, instance_, false);
        break;
      case 4:
        any = FormattedIntegerIO<4, DIR>(io_, instance_, false);
        break;
      case 8:
        any = FormattedIntegerIO<8, DIR>(io_, instance_, false);
        break;
      case 16:
        any = FormattedIntegerIO<16, DIR>(io_, instance_, false);
        break;
      default:
        handler.Crash(
            "not yet implemented: UNSIGNED(KIND=%d) in formatted IO", kind);
        return IostatEnd;
      }
      break;
    case TypeCategory::Real:
      switch (kind) {
      case 2:
        any = FormattedRealIO<2, DIR>(io_, instance_);
        break;
      case 3:
        any = FormattedRealIO<3, DIR>(io_, instance_);
        break;
      case 4:
        any = FormattedRealIO<4, DIR>(io_, instance_);
        break;
      case 8:
        any = FormattedRealIO<8, DIR>(io_, instance_);
        break;
      case 10:
        any = FormattedRealIO<10, DIR>(io_, instance_);
        break;
      // TODO: case double/double
      case 16:
        any = FormattedRealIO<16, DIR>(io_, instance_);
        break;
      default:
        handler.Crash(
            "not yet implemented: REAL(KIND=%d) in formatted IO", kind);
        return IostatEnd;
      }
      break;
    case TypeCategory::Complex:
      switch (kind) {
      case 2:
        any = FormattedComplexIO<2, DIR>(io_, instance_);
        break;
      case 3:
        any = FormattedComplexIO<3, DIR>(io_, instance_);
        break;
      case 4:
        any = FormattedComplexIO<4, DIR>(io_, instance_);
        break;
      case 8:
        any = FormattedComplexIO<8, DIR>(io_, instance_);
        break;
      case 10:
        any = FormattedComplexIO<10, DIR>(io_, instance_);
        break;
      // TODO: case double/double
      case 16:
        any = FormattedComplexIO<16, DIR>(io_, instance_);
        break;
      default:
        handler.Crash(
            "not yet implemented: COMPLEX(KIND=%d) in formatted IO", kind);
        return IostatEnd;
      }
      break;
    case TypeCategory::Character:
      switch (kind) {
      case 1:
        any = FormattedCharacterIO<char, DIR>(io_, instance_);
        break;
      case 2:
        any = FormattedCharacterIO<char16_t, DIR>(io_, instance_);
        break;
      case 4:
        any = FormattedCharacterIO<char32_t, DIR>(io_, instance_);
        break;
      default:
        handler.Crash(
            "not yet implemented: CHARACTER(KIND=%d) in formatted IO", kind);
        return IostatEnd;
      }
      break;
    case TypeCategory::Logical:
      switch (kind) {
      case 1:
        any = FormattedLogicalIO<1, DIR>(io_, instance_);
        break;
      case 2:
        any = FormattedLogicalIO<2, DIR>(io_, instance_);
        break;
      case 4:
        any = FormattedLogicalIO<4, DIR>(io_, instance_);
        break;
      case 8:
        any = FormattedLogicalIO<8, DIR>(io_, instance_);
        break;
      default:
        handler.Crash(
            "not yet implemented: LOGICAL(KIND=%d) in formatted IO", kind);
        return IostatEnd;
      }
      break;
    case TypeCategory::Derived: {
      // Derived type information must be present for formatted I/O.
      IoErrorHandler &handler{io_.GetIoErrorHandler()};
      const DescriptorAddendum *addendum{instance_.Addendum()};
      RUNTIME_CHECK(handler, addendum != nullptr);
      derived_ = addendum->derivedType();
      RUNTIME_CHECK(handler, derived_ != nullptr);
      if (table_) {
        if (const auto *definedIo{table_->Find(*derived_,
                DIR == Direction::Input ? common::DefinedIo::ReadFormatted
                                        : common::DefinedIo::WriteFormatted)}) {
          if (definedIo->subroutine) {
            nonTbpSpecial_.emplace(DIR == Direction::Input
                    ? typeInfo::SpecialBinding::Which::ReadFormatted
                    : typeInfo::SpecialBinding::Which::WriteFormatted,
                definedIo->subroutine,
                /*isArgDescriptorSet=*/
                (definedIo->flags & IsDtvArgPolymorphic) ? 1 : 0,
                /*isTypeBound=*/false,
                /*specialCaseFlag=*/!!(definedIo->flags & DefinedIoInteger8));
            special_ = &*nonTbpSpecial_;
          }
        }
      }
      if (!special_) {
        if (const typeInfo::SpecialBinding *binding{
                derived_->FindSpecialBinding(DIR == Direction::Input
                        ? typeInfo::SpecialBinding::Which::ReadFormatted
                        : typeInfo::SpecialBinding::Which::WriteFormatted)}) {
          if (!table_ || !table_->ignoreNonTbpEntries ||
              binding->IsTypeBound()) {
            special_ = binding;
          }
        }
      }
      return StatContinue;
    }
    }
    if (any) {
      anyIoTookPlace_ = true;
    } else {
      return IostatEnd;
    }
  } else {
    handler.Crash("DescriptorIO: bad type code (%d) in descriptor",
        static_cast<int>(instance_.type().raw()));
    return handler.GetIoStat();
  }
  return StatOk;
}

template RT_API_ATTRS int DescriptorIoTicket<Direction::Output>::Begin(
    WorkQueue &);
template RT_API_ATTRS int DescriptorIoTicket<Direction::Input>::Begin(
    WorkQueue &);

template <Direction DIR>
RT_API_ATTRS int DescriptorIoTicket<DIR>::Continue(WorkQueue &workQueue) {
  // Only derived type formatted I/O gets here.
  while (!IsComplete()) {
    if (special_) {
      if (auto defined{DefinedFormattedIo(
              io_, instance_, *derived_, *special_, subscripts_)}) {
        anyIoTookPlace_ |= *defined;
        Advance();
        continue;
      }
    }
    Descriptor &elementDesc{elementDescriptor_.descriptor()};
    elementDesc.Establish(
        *derived_, nullptr, 0, nullptr, CFI_attribute_pointer);
    elementDesc.set_base_addr(instance_.Element<char>(subscripts_));
    Advance();
    if (int status{workQueue.BeginDerivedIo<DIR>(
            io_, elementDesc, *derived_, table_, anyIoTookPlace_)};
        status != StatOk) {
      return status;
    }
  }
  return StatOk;
}

template RT_API_ATTRS int DescriptorIoTicket<Direction::Output>::Continue(
    WorkQueue &);
template RT_API_ATTRS int DescriptorIoTicket<Direction::Input>::Continue(
    WorkQueue &);

template <Direction DIR>
RT_API_ATTRS bool DescriptorIO(IoStatementState &io,
    const Descriptor &descriptor, const NonTbpDefinedIoTable *originalTable) {
  bool anyIoTookPlace{false};
  const NonTbpDefinedIoTable *defaultTable{io.nonTbpDefinedIoTable()};
  const NonTbpDefinedIoTable *table{originalTable};
  if (!table) {
    table = defaultTable;
  } else if (table != defaultTable) {
    io.set_nonTbpDefinedIoTable(table); // for nested I/O
  }
  WorkQueue workQueue{io.GetIoErrorHandler()};
  if (workQueue.BeginDescriptorIo<DIR>(io, descriptor, table, anyIoTookPlace) ==
      StatContinue) {
    workQueue.Run();
  }
  if (defaultTable != table) {
    io.set_nonTbpDefinedIoTable(defaultTable);
  }
  return anyIoTookPlace;
}

template RT_API_ATTRS bool DescriptorIO<Direction::Output>(
    IoStatementState &, const Descriptor &, const NonTbpDefinedIoTable *);
template RT_API_ATTRS bool DescriptorIO<Direction::Input>(
    IoStatementState &, const Descriptor &, const NonTbpDefinedIoTable *);

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime::io::descr
