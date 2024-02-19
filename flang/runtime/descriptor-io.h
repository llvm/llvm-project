//===-- runtime/descriptor-io.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_DESCRIPTOR_IO_H_
#define FORTRAN_RUNTIME_DESCRIPTOR_IO_H_

// Implementation of I/O data list item transfers based on descriptors.
// (All I/O items come through here so that the code is exercised for test;
// some scalar I/O data transfer APIs could be changed to bypass their use
// of descriptors in the future for better efficiency.)

#include "edit-input.h"
#include "edit-output.h"
#include "io-stmt.h"
#include "namelist.h"
#include "terminator.h"
#include "type-info.h"
#include "unit.h"
#include "flang/Common/uint128.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime::io::descr {
template <typename A>
inline A &ExtractElement(IoStatementState &io, const Descriptor &descriptor,
    const SubscriptValue subscripts[]) {
  A *p{descriptor.Element<A>(subscripts)};
  if (!p) {
    io.GetIoErrorHandler().Crash("Bad address for I/O item -- null base "
                                 "address or subscripts out of range");
  }
  return *p;
}

// Per-category descriptor-based I/O templates

// TODO (perhaps as a nontrivial but small starter project): implement
// automatic repetition counts, like "10*3.14159", for list-directed and
// NAMELIST array output.

template <int KIND, Direction DIR>
inline bool FormattedIntegerIO(
    IoStatementState &io, const Descriptor &descriptor) {
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  using IntType = CppTypeFor<TypeCategory::Integer, KIND>;
  bool anyInput{false};
  for (std::size_t j{0}; j < numElements; ++j) {
    if (auto edit{io.GetNextDataEdit()}) {
      IntType &x{ExtractElement<IntType>(io, descriptor, subscripts)};
      if constexpr (DIR == Direction::Output) {
        if (!EditIntegerOutput<KIND>(io, *edit, x)) {
          return false;
        }
      } else if (edit->descriptor != DataEdit::ListDirectedNullValue) {
        if (EditIntegerInput(io, *edit, reinterpret_cast<void *>(&x), KIND)) {
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
inline bool FormattedRealIO(
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
inline bool FormattedComplexIO(
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
inline bool FormattedCharacterIO(
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
inline bool FormattedLogicalIO(
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
static bool DescriptorIO(IoStatementState &, const Descriptor &,
    const NonTbpDefinedIoTable * = nullptr);

// For intrinsic (not defined) derived type I/O, formatted & unformatted
template <Direction DIR>
static bool DefaultComponentIO(IoStatementState &io,
    const typeInfo::Component &component, const Descriptor &origDescriptor,
    const SubscriptValue origSubscripts[], Terminator &terminator,
    const NonTbpDefinedIoTable *table) {
  if (component.genre() == typeInfo::Component::Genre::Data) {
    // Create a descriptor for the component
    StaticDescriptor<maxRank, true, 16 /*?*/> statDesc;
    Descriptor &desc{statDesc.descriptor()};
    component.CreatePointerDescriptor(
        desc, origDescriptor, terminator, origSubscripts);
    return DescriptorIO<DIR>(io, desc, table);
  } else {
    // Component is itself a descriptor
    char *pointer{
        origDescriptor.Element<char>(origSubscripts) + component.offset()};
    RUNTIME_CHECK(
        terminator, component.genre() == typeInfo::Component::Genre::Automatic);
    const Descriptor &compDesc{*reinterpret_cast<const Descriptor *>(pointer)};
    return DescriptorIO<DIR>(io, compDesc, table);
  }
}

template <Direction DIR>
static bool DefaultComponentwiseFormattedIO(IoStatementState &io,
    const Descriptor &descriptor, const typeInfo::DerivedType &type,
    const NonTbpDefinedIoTable *table, const SubscriptValue subscripts[]) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  const Descriptor &compArray{type.component()};
  RUNTIME_CHECK(handler, compArray.rank() == 1);
  std::size_t numComponents{compArray.Elements()};
  SubscriptValue at[maxRank];
  compArray.GetLowerBounds(at);
  for (std::size_t k{0}; k < numComponents;
       ++k, compArray.IncrementSubscripts(at)) {
    const typeInfo::Component &component{
        *compArray.Element<typeInfo::Component>(at)};
    if (!DefaultComponentIO<DIR>(
            io, component, descriptor, subscripts, handler, table)) {
      // Return true for NAMELIST input if any component appeared.
      auto *listInput{
          io.get_if<ListDirectedStatementState<Direction::Input>>()};
      return DIR == Direction::Input && k > 0 && listInput &&
          listInput->inNamelistSequence();
    }
  }
  return true;
}

template <Direction DIR>
static bool DefaultComponentwiseUnformattedIO(IoStatementState &io,
    const Descriptor &descriptor, const typeInfo::DerivedType &type,
    const NonTbpDefinedIoTable *table) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  const Descriptor &compArray{type.component()};
  RUNTIME_CHECK(handler, compArray.rank() == 1);
  std::size_t numComponents{compArray.Elements()};
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  for (std::size_t j{0}; j < numElements;
       ++j, descriptor.IncrementSubscripts(subscripts)) {
    SubscriptValue at[maxRank];
    compArray.GetLowerBounds(at);
    for (std::size_t k{0}; k < numComponents;
         ++k, compArray.IncrementSubscripts(at)) {
      const typeInfo::Component &component{
          *compArray.Element<typeInfo::Component>(at)};
      if (!DefaultComponentIO<DIR>(
              io, component, descriptor, subscripts, handler, table)) {
        return false;
      }
    }
  }
  return true;
}

std::optional<bool> DefinedFormattedIo(IoStatementState &, const Descriptor &,
    const typeInfo::DerivedType &, const typeInfo::SpecialBinding &,
    const SubscriptValue[]);

template <Direction DIR>
static bool FormattedDerivedTypeIO(IoStatementState &io,
    const Descriptor &descriptor, const NonTbpDefinedIoTable *table) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  // Derived type information must be present for formatted I/O.
  const DescriptorAddendum *addendum{descriptor.Addendum()};
  RUNTIME_CHECK(handler, addendum != nullptr);
  const typeInfo::DerivedType *type{addendum->derivedType()};
  RUNTIME_CHECK(handler, type != nullptr);
  std::optional<typeInfo::SpecialBinding> nonTbpSpecial;
  const typeInfo::SpecialBinding *special{nullptr};
  if (table) {
    if (const auto *definedIo{table->Find(*type,
            DIR == Direction::Input ? common::DefinedIo::ReadFormatted
                                    : common::DefinedIo::WriteFormatted)}) {
      if (definedIo->subroutine) {
        nonTbpSpecial.emplace(DIR == Direction::Input
                ? typeInfo::SpecialBinding::Which::ReadFormatted
                : typeInfo::SpecialBinding::Which::WriteFormatted,
            definedIo->subroutine, definedIo->isDtvArgPolymorphic, false,
            false);
        special = &*nonTbpSpecial;
      }
    }
  }
  if (!special) {
    if (const typeInfo::SpecialBinding *
        binding{type->FindSpecialBinding(DIR == Direction::Input
                ? typeInfo::SpecialBinding::Which::ReadFormatted
                : typeInfo::SpecialBinding::Which::WriteFormatted)}) {
      if (!table || !table->ignoreNonTbpEntries || binding->isTypeBound()) {
        special = binding;
      }
    }
  }
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  std::size_t numElements{descriptor.Elements()};
  for (std::size_t j{0}; j < numElements;
       ++j, descriptor.IncrementSubscripts(subscripts)) {
    std::optional<bool> result;
    if (special) {
      result = DefinedFormattedIo(io, descriptor, *type, *special, subscripts);
    }
    if (!result) {
      result = DefaultComponentwiseFormattedIO<DIR>(
          io, descriptor, *type, table, subscripts);
    }
    if (!result.value()) {
      // Return true for NAMELIST input if we got anything.
      auto *listInput{
          io.get_if<ListDirectedStatementState<Direction::Input>>()};
      return DIR == Direction::Input && j > 0 && listInput &&
          listInput->inNamelistSequence();
    }
  }
  return true;
}

bool DefinedUnformattedIo(IoStatementState &, const Descriptor &,
    const typeInfo::DerivedType &, const typeInfo::SpecialBinding &);

// Unformatted I/O
template <Direction DIR>
static bool UnformattedDescriptorIO(IoStatementState &io,
    const Descriptor &descriptor, const NonTbpDefinedIoTable *table = nullptr) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  const DescriptorAddendum *addendum{descriptor.Addendum()};
  if (const typeInfo::DerivedType *
      type{addendum ? addendum->derivedType() : nullptr}) {
    // derived type unformatted I/O
    if (table) {
      if (const auto *definedIo{table->Find(*type,
              DIR == Direction::Input ? common::DefinedIo::ReadUnformatted
                                      : common::DefinedIo::WriteUnformatted)}) {
        if (definedIo->subroutine) {
          typeInfo::SpecialBinding special{DIR == Direction::Input
                  ? typeInfo::SpecialBinding::Which::ReadUnformatted
                  : typeInfo::SpecialBinding::Which::WriteUnformatted,
              definedIo->subroutine, definedIo->isDtvArgPolymorphic, false,
              false};
          if (std::optional<bool> wasDefined{
                  DefinedUnformattedIo(io, descriptor, *type, special)}) {
            return *wasDefined;
          }
        } else {
          return DefaultComponentwiseUnformattedIO<DIR>(
              io, descriptor, *type, table);
        }
      }
    }
    if (const typeInfo::SpecialBinding *
        special{type->FindSpecialBinding(DIR == Direction::Input
                ? typeInfo::SpecialBinding::Which::ReadUnformatted
                : typeInfo::SpecialBinding::Which::WriteUnformatted)}) {
      if (!table || !table->ignoreNonTbpEntries || special->isTypeBound()) {
        // defined derived type unformatted I/O
        return DefinedUnformattedIo(io, descriptor, *type, *special);
      }
    }
    // Default derived type unformatted I/O
    // TODO: If no component at any level has defined READ or WRITE
    // (as appropriate), the elements are contiguous, and no byte swapping
    // is active, do a block transfer via the code below.
    return DefaultComponentwiseUnformattedIO<DIR>(io, descriptor, *type, table);
  } else {
    // intrinsic type unformatted I/O
    auto *externalUnf{io.get_if<ExternalUnformattedIoStatementState<DIR>>()};
    auto *childUnf{io.get_if<ChildUnformattedIoStatementState<DIR>>()};
    auto *inq{
        DIR == Direction::Output ? io.get_if<InquireIOLengthState>() : nullptr};
    RUNTIME_CHECK(handler, externalUnf || childUnf || inq);
    std::size_t elementBytes{descriptor.ElementBytes()};
    std::size_t numElements{descriptor.Elements()};
    std::size_t swappingBytes{elementBytes};
    if (auto maybeCatAndKind{descriptor.type().GetCategoryAndKind()}) {
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
    SubscriptValue subscripts[maxRank];
    descriptor.GetLowerBounds(subscripts);
    using CharType =
        std::conditional_t<DIR == Direction::Output, const char, char>;
    auto Transfer{[=](CharType &x, std::size_t totalBytes) -> bool {
      if constexpr (DIR == Direction::Output) {
        return externalUnf ? externalUnf->Emit(&x, totalBytes, swappingBytes)
            : childUnf     ? childUnf->Emit(&x, totalBytes, swappingBytes)
                           : inq->Emit(&x, totalBytes, swappingBytes);
      } else {
        return externalUnf ? externalUnf->Receive(&x, totalBytes, swappingBytes)
                           : childUnf->Receive(&x, totalBytes, swappingBytes);
      }
    }};
    bool swapEndianness{externalUnf && externalUnf->unit().swapEndianness()};
    if (!swapEndianness &&
        descriptor.IsContiguous()) { // contiguous unformatted I/O
      char &x{ExtractElement<char>(io, descriptor, subscripts)};
      return Transfer(x, numElements * elementBytes);
    } else { // non-contiguous or byte-swapped intrinsic type unformatted I/O
      for (std::size_t j{0}; j < numElements; ++j) {
        char &x{ExtractElement<char>(io, descriptor, subscripts)};
        if (!Transfer(x, elementBytes)) {
          return false;
        }
        if (!descriptor.IncrementSubscripts(subscripts) &&
            j + 1 < numElements) {
          handler.Crash("DescriptorIO: subscripts out of bounds");
        }
      }
      return true;
    }
  }
}

template <Direction DIR>
static bool DescriptorIO(IoStatementState &io, const Descriptor &descriptor,
    const NonTbpDefinedIoTable *table) {
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  if (handler.InError()) {
    return false;
  }
  if (!io.get_if<IoDirectionState<DIR>>()) {
    io.GetIoErrorHandler().Crash(
        "DescriptorIO() called for wrong I/O direction");
    return false;
  }
  if constexpr (DIR == Direction::Input) {
    if (!io.BeginReadingRecord()) {
      return false;
    }
  }
  if (!io.get_if<FormattedIoStatementState<DIR>>()) {
    return UnformattedDescriptorIO<DIR>(io, descriptor, table);
  }
  if (auto catAndKind{descriptor.type().GetCategoryAndKind()}) {
    TypeCategory cat{catAndKind->first};
    int kind{catAndKind->second};
    switch (cat) {
    case TypeCategory::Integer:
      switch (kind) {
      case 1:
        return FormattedIntegerIO<1, DIR>(io, descriptor);
      case 2:
        return FormattedIntegerIO<2, DIR>(io, descriptor);
      case 4:
        return FormattedIntegerIO<4, DIR>(io, descriptor);
      case 8:
        return FormattedIntegerIO<8, DIR>(io, descriptor);
      case 16:
        return FormattedIntegerIO<16, DIR>(io, descriptor);
      default:
        handler.Crash(
            "not yet implemented: INTEGER(KIND=%d) in formatted IO", kind);
        return false;
      }
    case TypeCategory::Real:
      switch (kind) {
      case 2:
        return FormattedRealIO<2, DIR>(io, descriptor);
      case 3:
        return FormattedRealIO<3, DIR>(io, descriptor);
      case 4:
        return FormattedRealIO<4, DIR>(io, descriptor);
      case 8:
        return FormattedRealIO<8, DIR>(io, descriptor);
      case 10:
        return FormattedRealIO<10, DIR>(io, descriptor);
      // TODO: case double/double
      case 16:
        return FormattedRealIO<16, DIR>(io, descriptor);
      default:
        handler.Crash(
            "not yet implemented: REAL(KIND=%d) in formatted IO", kind);
        return false;
      }
    case TypeCategory::Complex:
      switch (kind) {
      case 2:
        return FormattedComplexIO<2, DIR>(io, descriptor);
      case 3:
        return FormattedComplexIO<3, DIR>(io, descriptor);
      case 4:
        return FormattedComplexIO<4, DIR>(io, descriptor);
      case 8:
        return FormattedComplexIO<8, DIR>(io, descriptor);
      case 10:
        return FormattedComplexIO<10, DIR>(io, descriptor);
      // TODO: case double/double
      case 16:
        return FormattedComplexIO<16, DIR>(io, descriptor);
      default:
        handler.Crash(
            "not yet implemented: COMPLEX(KIND=%d) in formatted IO", kind);
        return false;
      }
    case TypeCategory::Character:
      switch (kind) {
      case 1:
        return FormattedCharacterIO<char, DIR>(io, descriptor);
      case 2:
        return FormattedCharacterIO<char16_t, DIR>(io, descriptor);
      case 4:
        return FormattedCharacterIO<char32_t, DIR>(io, descriptor);
      default:
        handler.Crash(
            "not yet implemented: CHARACTER(KIND=%d) in formatted IO", kind);
        return false;
      }
    case TypeCategory::Logical:
      switch (kind) {
      case 1:
        return FormattedLogicalIO<1, DIR>(io, descriptor);
      case 2:
        return FormattedLogicalIO<2, DIR>(io, descriptor);
      case 4:
        return FormattedLogicalIO<4, DIR>(io, descriptor);
      case 8:
        return FormattedLogicalIO<8, DIR>(io, descriptor);
      default:
        handler.Crash(
            "not yet implemented: LOGICAL(KIND=%d) in formatted IO", kind);
        return false;
      }
    case TypeCategory::Derived:
      return FormattedDerivedTypeIO<DIR>(io, descriptor, table);
    }
  }
  handler.Crash("DescriptorIO: bad type code (%d) in descriptor",
      static_cast<int>(descriptor.type().raw()));
  return false;
}
} // namespace Fortran::runtime::io::descr
#endif // FORTRAN_RUNTIME_DESCRIPTOR_IO_H_
