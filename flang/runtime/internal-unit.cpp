//===-- runtime/internal-unit.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "internal-unit.h"
#include "descriptor.h"
#include "io-error.h"
#include <algorithm>
#include <type_traits>

namespace Fortran::runtime::io {

template <Direction DIR>
InternalDescriptorUnit<DIR>::InternalDescriptorUnit(
    Scalar scalar, std::size_t length) {
  isFixedRecordLength = true;
  recordLength = length;
  endfileRecordNumber = 2;
  void *pointer{reinterpret_cast<void *>(const_cast<char *>(scalar))};
  descriptor().Establish(TypeCode{CFI_type_char}, length, pointer, 0, nullptr,
      CFI_attribute_pointer);
}

template <Direction DIR>
InternalDescriptorUnit<DIR>::InternalDescriptorUnit(
    const Descriptor &that, const Terminator &terminator) {
  RUNTIME_CHECK(terminator, that.type().IsCharacter());
  Descriptor &d{descriptor()};
  RUNTIME_CHECK(
      terminator, that.SizeInBytes() <= d.SizeInBytes(maxRank, true, 0));
  new (&d) Descriptor{that};
  d.Check();
  isFixedRecordLength = true;
  recordLength = d.ElementBytes();
  endfileRecordNumber = d.Elements() + 1;
}

template <Direction DIR> void InternalDescriptorUnit<DIR>::EndIoStatement() {
  if constexpr (DIR == Direction::Output) { // blank fill
    while (char *record{CurrentRecord()}) {
      if (furthestPositionInRecord <
          recordLength.value_or(furthestPositionInRecord)) {
        std::fill_n(record + furthestPositionInRecord,
            *recordLength - furthestPositionInRecord, ' ');
      }
      furthestPositionInRecord = 0;
      ++currentRecordNumber;
    }
  }
}

template <Direction DIR>
bool InternalDescriptorUnit<DIR>::Emit(
    const char *data, std::size_t bytes, IoErrorHandler &handler) {
  if constexpr (DIR == Direction::Input) {
    handler.Crash("InternalDescriptorUnit<Direction::Input>::Emit() called");
    return false && data[bytes] != 0; // bogus compare silences GCC warning
  } else {
    if (bytes <= 0) {
      return true;
    }
    char *record{CurrentRecord()};
    if (!record) {
      handler.SignalError(IostatInternalWriteOverrun);
      return false;
    }
    auto furthestAfter{std::max(furthestPositionInRecord,
        positionInRecord + static_cast<std::int64_t>(bytes))};
    bool ok{true};
    if (furthestAfter > static_cast<std::int64_t>(recordLength.value_or(0))) {
      handler.SignalError(IostatRecordWriteOverrun);
      furthestAfter = recordLength.value_or(0);
      bytes = std::max(std::int64_t{0}, furthestAfter - positionInRecord);
      ok = false;
    } else if (positionInRecord > furthestPositionInRecord) {
      std::fill_n(record + furthestPositionInRecord,
          positionInRecord - furthestPositionInRecord, ' ');
    }
    std::memcpy(record + positionInRecord, data, bytes);
    positionInRecord += bytes;
    furthestPositionInRecord = furthestAfter;
    return ok;
  }
}

template <Direction DIR>
std::optional<char32_t> InternalDescriptorUnit<DIR>::GetCurrentChar(
    IoErrorHandler &handler) {
  if constexpr (DIR == Direction::Output) {
    handler.Crash(
        "InternalDescriptorUnit<Direction::Output>::GetCurrentChar() called");
    return std::nullopt;
  }
  const char *record{CurrentRecord()};
  if (!record) {
    handler.SignalEnd();
    return std::nullopt;
  }
  if (positionInRecord >= recordLength.value_or(positionInRecord)) {
    return std::nullopt;
  }
  if (isUTF8) {
    // TODO: UTF-8 decoding
  }
  return record[positionInRecord];
}

template <Direction DIR>
bool InternalDescriptorUnit<DIR>::AdvanceRecord(IoErrorHandler &handler) {
  if (currentRecordNumber >= endfileRecordNumber.value_or(0)) {
    handler.SignalEnd();
    return false;
  }
  if constexpr (DIR == Direction::Output) { // blank fill
    if (furthestPositionInRecord <
        recordLength.value_or(furthestPositionInRecord)) {
      char *record{CurrentRecord()};
      RUNTIME_CHECK(handler, record != nullptr);
      std::fill_n(record + furthestPositionInRecord,
          *recordLength - furthestPositionInRecord, ' ');
    }
  }
  ++currentRecordNumber;
  BeginRecord();
  return true;
}

template <Direction DIR>
void InternalDescriptorUnit<DIR>::BackspaceRecord(IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, currentRecordNumber > 1);
  --currentRecordNumber;
  BeginRecord();
}

template class InternalDescriptorUnit<Direction::Output>;
template class InternalDescriptorUnit<Direction::Input>;
} // namespace Fortran::runtime::io
