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

#include "cpp-type.h"
#include "descriptor.h"
#include "edit-input.h"
#include "edit-output.h"
#include "io-stmt.h"
#include "terminator.h"
#include "flang/Common/uint128.h"

namespace Fortran::runtime::io::descr {
template <typename A>
inline A &ExtractElement(IoStatementState &io, const Descriptor &descriptor,
    const SubscriptValue subscripts[]) {
  A *p{descriptor.Element<A>(subscripts)};
  if (!p) {
    io.GetIoErrorHandler().Crash("ExtractElement: subscripts out of range");
  }
  return *p;
}

// Per-category descriptor-based I/O templates

// TODO (perhaps as a nontrivial but small starter project): implement
// automatic repetition counts, like "10*3.14159", for list-directed and
// NAMELIST array output.

template <typename A, Direction DIR>
inline bool FormattedIntegerIO(
    IoStatementState &io, const Descriptor &descriptor) {
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  for (std::size_t j{0}; j < numElements; ++j) {
    if (auto edit{io.GetNextDataEdit()}) {
      A &x{ExtractElement<A>(io, descriptor, subscripts)};
      if constexpr (DIR == Direction::Output) {
        if (!EditIntegerOutput(io, *edit, static_cast<std::int64_t>(x))) {
          return false;
        }
      } else if (edit->descriptor != DataEdit::ListDirectedNullValue) {
        if (!EditIntegerInput(io, *edit, reinterpret_cast<void *>(&x),
                static_cast<int>(sizeof(A)))) {
          return false;
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
  for (std::size_t j{0}; j < numElements; ++j) {
    if (auto edit{io.GetNextDataEdit()}) {
      RawType &x{ExtractElement<RawType>(io, descriptor, subscripts)};
      if constexpr (DIR == Direction::Output) {
        if (!RealOutputEditing<KIND>{io, x}.Edit(*edit)) {
          return false;
        }
      } else if (edit->descriptor != DataEdit::ListDirectedNullValue) {
        if (!EditRealInput<KIND>(io, *edit, reinterpret_cast<void *>(&x))) {
          return false;
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
  for (std::size_t j{0}; j < numElements; ++j) {
    RawType *x{&ExtractElement<RawType>(io, descriptor, subscripts)};
    if (isListOutput) {
      DataEdit rEdit, iEdit;
      rEdit.descriptor = DataEdit::ListDirectedRealPart;
      iEdit.descriptor = DataEdit::ListDirectedImaginaryPart;
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
        } else if (!EditRealInput<KIND>(
                       io, *edit, reinterpret_cast<void *>(x))) {
          return false;
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
  for (std::size_t j{0}; j < numElements; ++j) {
    A *x{&ExtractElement<A>(io, descriptor, subscripts)};
    if (listOutput) {
      if (!ListDirectedDefaultCharacterOutput(io, *listOutput, x, length)) {
        return false;
      }
    } else if (auto edit{io.GetNextDataEdit()}) {
      if constexpr (DIR == Direction::Output) {
        if (!EditDefaultCharacterOutput(io, *edit, x, length)) {
          return false;
        }
      } else {
        if (edit->descriptor != DataEdit::ListDirectedNullValue) {
          if (!EditDefaultCharacterInput(io, *edit, x, length)) {
            return false;
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

template <typename A, Direction DIR>
inline bool FormattedLogicalIO(
    IoStatementState &io, const Descriptor &descriptor) {
  std::size_t numElements{descriptor.Elements()};
  SubscriptValue subscripts[maxRank];
  descriptor.GetLowerBounds(subscripts);
  auto *listOutput{io.get_if<ListDirectedStatementState<Direction::Output>>()};
  for (std::size_t j{0}; j < numElements; ++j) {
    A &x{ExtractElement<A>(io, descriptor, subscripts)};
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
          } else {
            return false;
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
static bool DescriptorIO(IoStatementState &io, const Descriptor &descriptor) {
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
  if (auto *unf{io.get_if<UnformattedIoStatementState<DIR>>()}) {
    std::size_t elementBytes{descriptor.ElementBytes()};
    SubscriptValue subscripts[maxRank];
    descriptor.GetLowerBounds(subscripts);
    std::size_t numElements{descriptor.Elements()};
    if (descriptor.IsContiguous()) { // contiguous unformatted I/O
      char &x{ExtractElement<char>(io, descriptor, subscripts)};
      auto totalBytes{numElements * elementBytes};
      if constexpr (DIR == Direction::Output) {
        return unf->Emit(&x, totalBytes, elementBytes);
      } else {
        return unf->Receive(&x, totalBytes, elementBytes);
      }
    } else { // non-contiguous unformatted I/O
      for (std::size_t j{0}; j < numElements; ++j) {
        char &x{ExtractElement<char>(io, descriptor, subscripts)};
        if constexpr (DIR == Direction::Output) {
          if (!unf->Emit(&x, elementBytes, elementBytes)) {
            return false;
          }
        } else {
          if (!unf->Receive(&x, elementBytes, elementBytes)) {
            return false;
          }
        }
        if (!descriptor.IncrementSubscripts(subscripts) &&
            j + 1 < numElements) {
          io.GetIoErrorHandler().Crash(
              "DescriptorIO: subscripts out of bounds");
        }
      }
      return true;
    }
  } else if (auto catAndKind{descriptor.type().GetCategoryAndKind()}) {
    int kind{catAndKind->second};
    switch (catAndKind->first) {
    case TypeCategory::Integer:
      switch (kind) {
      case 1:
        return FormattedIntegerIO<CppTypeFor<TypeCategory::Integer, 1>, DIR>(
            io, descriptor);
      case 2:
        return FormattedIntegerIO<CppTypeFor<TypeCategory::Integer, 2>, DIR>(
            io, descriptor);
      case 4:
        return FormattedIntegerIO<CppTypeFor<TypeCategory::Integer, 4>, DIR>(
            io, descriptor);
      case 8:
        return FormattedIntegerIO<CppTypeFor<TypeCategory::Integer, 8>, DIR>(
            io, descriptor);
      case 16:
        return FormattedIntegerIO<CppTypeFor<TypeCategory::Integer, 16>, DIR>(
            io, descriptor);
      default:
        io.GetIoErrorHandler().Crash(
            "DescriptorIO: Unimplemented INTEGER kind (%d) in descriptor",
            kind);
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
        io.GetIoErrorHandler().Crash(
            "DescriptorIO: Unimplemented REAL kind (%d) in descriptor", kind);
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
        io.GetIoErrorHandler().Crash(
            "DescriptorIO: Unimplemented COMPLEX kind (%d) in descriptor",
            kind);
        return false;
      }
    case TypeCategory::Character:
      switch (kind) {
      case 1:
        return FormattedCharacterIO<char, DIR>(io, descriptor);
      // TODO cases 2, 4
      default:
        io.GetIoErrorHandler().Crash(
            "DescriptorIO: Unimplemented CHARACTER kind (%d) in descriptor",
            kind);
        return false;
      }
    case TypeCategory::Logical:
      switch (kind) {
      case 1:
        return FormattedLogicalIO<CppTypeFor<TypeCategory::Integer, 1>, DIR>(
            io, descriptor);
      case 2:
        return FormattedLogicalIO<CppTypeFor<TypeCategory::Integer, 2>, DIR>(
            io, descriptor);
      case 4:
        return FormattedLogicalIO<CppTypeFor<TypeCategory::Integer, 4>, DIR>(
            io, descriptor);
      case 8:
        return FormattedLogicalIO<CppTypeFor<TypeCategory::Integer, 8>, DIR>(
            io, descriptor);
      default:
        io.GetIoErrorHandler().Crash(
            "DescriptorIO: Unimplemented LOGICAL kind (%d) in descriptor",
            kind);
        return false;
      }
    case TypeCategory::Derived:
      io.GetIoErrorHandler().Crash(
          "DescriptorIO: Unimplemented: derived type I/O",
          static_cast<int>(descriptor.type().raw()));
      return false;
    }
  }
  io.GetIoErrorHandler().Crash("DescriptorIO: Bad type code (%d) in descriptor",
      static_cast<int>(descriptor.type().raw()));
  return false;
}
} // namespace Fortran::runtime::io::descr
#endif // FORTRAN_RUNTIME_DESCRIPTOR_IO_H_
