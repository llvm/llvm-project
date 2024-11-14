//===-- runtime/internal-unit.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "internal-unit.h"
#include "io-error.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/freestanding-tools.h"
#include <algorithm>
#include <type_traits>

namespace Fortran::runtime::io {
RT_OFFLOAD_API_GROUP_BEGIN

template <Direction DIR>
RT_API_ATTRS InternalDescriptorUnit<DIR>::InternalDescriptorUnit(
    Scalar scalar, std::size_t length, int kind) {
  internalIoCharKind = kind;
  recordLength = length;
  endfileRecordNumber = 2;
  void *pointer{reinterpret_cast<void *>(const_cast<char *>(scalar))};
  descriptor().Establish(TypeCode{TypeCategory::Character, kind}, length * kind,
      pointer, 0, nullptr, CFI_attribute_pointer);
}

template <Direction DIR>
RT_API_ATTRS InternalDescriptorUnit<DIR>::InternalDescriptorUnit(
    const Descriptor &that, const Terminator &terminator) {
  auto thatType{that.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, thatType.has_value());
  RUNTIME_CHECK(terminator, thatType->first == TypeCategory::Character);
  Descriptor &d{descriptor()};
  RUNTIME_CHECK(
      terminator, that.SizeInBytes() <= d.SizeInBytes(maxRank, true, 0));
  new (&d) Descriptor{that};
  d.Check();
  internalIoCharKind = thatType->second;
  recordLength = d.ElementBytes();
  endfileRecordNumber = d.Elements() + 1;
}

template <Direction DIR>
RT_API_ATTRS bool InternalDescriptorUnit<DIR>::Emit(
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
      BlankFill(record + furthestPositionInRecord,
          positionInRecord - furthestPositionInRecord);
    }
    std::memcpy(record + positionInRecord, data, bytes);
    positionInRecord += bytes;
    furthestPositionInRecord = furthestAfter;
    return ok;
  }
}

template <Direction DIR>
RT_API_ATTRS std::size_t InternalDescriptorUnit<DIR>::GetNextInputBytes(
    const char *&p, IoErrorHandler &handler) {
  p = nullptr;
  if constexpr (DIR == Direction::Output) {
    handler.Crash("InternalDescriptorUnit<Direction::Output>::"
                  "GetNextInputBytes() called");
    return 0;
  } else {
    const char *record{CurrentRecord()};
    if (!record) {
      handler.SignalEnd();
      return 0;
    } else if (positionInRecord >= recordLength.value_or(positionInRecord)) {
      return 0;
    } else {
      p = &record[positionInRecord];
      return *recordLength - positionInRecord;
    }
  }
}

template <Direction DIR>
RT_API_ATTRS std::size_t InternalDescriptorUnit<DIR>::ViewBytesInRecord(
    const char *&p, bool forward) const {
  p = nullptr;
  auto recl{recordLength.value_or(positionInRecord)};
  const char *record{CurrentRecord()};
  if (forward) {
    if (positionInRecord < recl) {
      if (record) {
        p = &record[positionInRecord];
      }
      return recl - positionInRecord;
    }
  } else {
    if (record && positionInRecord <= recl) {
      p = &record[positionInRecord];
    }
    return positionInRecord - leftTabLimit.value_or(0);
  }
  return 0;
}

template <Direction DIR>
RT_API_ATTRS bool InternalDescriptorUnit<DIR>::AdvanceRecord(
    IoErrorHandler &handler) {
  if (currentRecordNumber >= endfileRecordNumber.value_or(0)) {
    if constexpr (DIR == Direction::Input) {
      handler.SignalEnd();
    } else {
      handler.SignalError(IostatInternalWriteOverrun);
    }
    return false;
  }
  if constexpr (DIR == Direction::Output) {
    BlankFillOutputRecord();
  }
  ++currentRecordNumber;
  BeginRecord();
  return true;
}

template <Direction DIR>
RT_API_ATTRS void InternalDescriptorUnit<DIR>::BlankFill(
    char *at, std::size_t bytes) {
  switch (internalIoCharKind) {
  case 2:
    Fortran::runtime::fill_n(reinterpret_cast<char16_t *>(at), bytes / 2,
        static_cast<char16_t>(' '));
    break;
  case 4:
    Fortran::runtime::fill_n(reinterpret_cast<char32_t *>(at), bytes / 4,
        static_cast<char32_t>(' '));
    break;
  default:
    Fortran::runtime::fill_n(at, bytes, ' ');
    break;
  }
}

template <Direction DIR>
RT_API_ATTRS void InternalDescriptorUnit<DIR>::BlankFillOutputRecord() {
  if constexpr (DIR == Direction::Output) {
    if (furthestPositionInRecord <
        recordLength.value_or(furthestPositionInRecord)) {
      BlankFill(CurrentRecord() + furthestPositionInRecord,
          *recordLength - furthestPositionInRecord);
    }
  }
}

template <Direction DIR>
RT_API_ATTRS void InternalDescriptorUnit<DIR>::BackspaceRecord(
    IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, currentRecordNumber > 1);
  --currentRecordNumber;
  BeginRecord();
}

template <Direction DIR>
RT_API_ATTRS std::int64_t InternalDescriptorUnit<DIR>::InquirePos() {
  return (currentRecordNumber - 1) * recordLength.value_or(0) +
      positionInRecord + 1;
}

template class InternalDescriptorUnit<Direction::Output>;
template class InternalDescriptorUnit<Direction::Input>;

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime::io
