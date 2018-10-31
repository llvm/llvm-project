//===-- xray_fdr_log_writer.h ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a function call tracing system.
//
//===----------------------------------------------------------------------===//
#ifndef COMPILER_RT_LIB_XRAY_XRAY_FDR_LOG_WRITER_H_
#define COMPILER_RT_LIB_XRAY_XRAY_FDR_LOG_WRITER_H_

#include "xray_buffer_queue.h"
#include "xray_fdr_log_records.h"
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace __xray {

template <size_t Index> struct SerializerImpl {
  template <class Tuple,
            typename std::enable_if<
                Index<std::tuple_size<
                          typename std::remove_reference<Tuple>::type>::value,
                      int>::type = 0> static void serializeTo(char *Buffer,
                                                              Tuple &&T) {
    auto P = reinterpret_cast<const char *>(&std::get<Index>(T));
    constexpr auto Size = sizeof(std::get<Index>(T));
    internal_memcpy(Buffer, P, Size);
    SerializerImpl<Index + 1>::serializeTo(Buffer + Size,
                                           std::forward<Tuple>(T));
  }

  template <class Tuple,
            typename std::enable_if<
                Index >= std::tuple_size<typename std::remove_reference<
                             Tuple>::type>::value,
                int>::type = 0>
  static void serializeTo(char *, Tuple &&){};
};

using Serializer = SerializerImpl<0>;

template <MetadataRecord::RecordKinds Kind, class... DataTypes>
MetadataRecord createMetadataRecord(DataTypes &&... Ds) {
  MetadataRecord R;
  R.Type = 1;
  R.RecordKind = static_cast<uint8_t>(Kind);
  Serializer::serializeTo(R.Data,
                          std::make_tuple(std::forward<DataTypes>(Ds)...));
  return R;
}

class FDRLogWriter {
  BufferQueue::Buffer &Buffer;
  char *NextRecord = nullptr;

  template <class T> void writeRecord(const T &R) {
    internal_memcpy(NextRecord, reinterpret_cast<const char *>(&R), sizeof(T));
    NextRecord += sizeof(T);
    atomic_fetch_add(&Buffer.Extents, sizeof(T), memory_order_acq_rel);
  }

public:
  explicit FDRLogWriter(BufferQueue::Buffer &B, char *P)
      : Buffer(B), NextRecord(P) {
    DCHECK_NE(Buffer.Data, nullptr);
    DCHECK_NE(NextRecord, nullptr);
  }

  explicit FDRLogWriter(BufferQueue::Buffer &B)
      : FDRLogWriter(B, static_cast<char *>(B.Data)) {}

  template <MetadataRecord::RecordKinds Kind, class... Data>
  bool writeMetadata(Data &&... Ds) {
    // TODO: Check boundary conditions:
    // 1) Buffer is full, and cannot handle one metadata record.
    // 2) Buffer queue is finalising.
    writeRecord(createMetadataRecord<Kind>(std::forward<Data>(Ds)...));
    return true;
  }

  template <size_t N> size_t writeMetadataRecords(MetadataRecord (&Recs)[N]) {
    constexpr auto Size = sizeof(MetadataRecord) * N;
    internal_memcpy(NextRecord, reinterpret_cast<const char *>(Recs), Size);
    NextRecord += Size;
    atomic_fetch_add(&Buffer.Extents, Size, memory_order_acq_rel);
    return Size;
  }

  enum class FunctionRecordKind : uint8_t {
    Enter = 0x00,
    Exit = 0x01,
    TailExit = 0x02,
    EnterArg = 0x03,
  };

  bool writeFunction(FunctionRecordKind Kind, int32_t FuncId, int32_t Delta) {
    FunctionRecord R;
    R.Type = 0;
    R.RecordKind = uint8_t(Kind);
    R.FuncId = FuncId;
    R.TSCDelta = Delta;
    writeRecord(R);
    return true;
  }

  bool writeCustomEvent(uint64_t TSC, const void *Event, int32_t EventSize) {
    writeMetadata<MetadataRecord::RecordKinds::CustomEventMarker>(EventSize,
                                                                  TSC);
    internal_memcpy(NextRecord, Event, EventSize);
    NextRecord += EventSize;
    atomic_fetch_add(&Buffer.Extents, EventSize, memory_order_acq_rel);
    return true;
  }

  bool writeTypedEvent(uint64_t TSC, uint16_t EventType, const void *Event,
                       int32_t EventSize) {
    writeMetadata<MetadataRecord::RecordKinds::TypedEventMarker>(EventSize, TSC,
                                                                 EventType);
    internal_memcpy(NextRecord, Event, EventSize);
    NextRecord += EventSize;
    atomic_fetch_add(&Buffer.Extents, EventSize, memory_order_acq_rel);
    return true;
  }

  char *getNextRecord() const { return NextRecord; }

  void resetRecord() {
    NextRecord = reinterpret_cast<char *>(Buffer.Data);
    atomic_store(&Buffer.Extents, 0, memory_order_release);
  }

  void undoWrites(size_t B) {
    DCHECK_GE(NextRecord - B, reinterpret_cast<char *>(Buffer.Data));
    NextRecord -= B;
    atomic_fetch_sub(&Buffer.Extents, B, memory_order_acq_rel);
  }

}; // namespace __xray

} // namespace __xray

#endif // COMPILER-RT_LIB_XRAY_XRAY_FDR_LOG_WRITER_H_
