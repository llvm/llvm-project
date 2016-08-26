//===-- xray_records.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This header exposes some record types useful for the XRay in-memory logging
// implementation.
//
//===----------------------------------------------------------------------===//

#ifndef XRAY_XRAY_RECORDS_H
#define XRAY_XRAY_RECORDS_H

namespace __xray {

enum FileTypes {
  NAIVE_LOG = 0,
};

// This data structure is used to describe the contents of the file. We use this
// for versioning the supported XRay file formats.
struct alignas(32) XRayFileHeader {
  uint16_t Version = 0;

  // The type of file we're writing out. See the FileTypes enum for more
  // information. This allows different implementations of the XRay logging to
  // have different files for different information being stored.
  uint16_t Type = 0;

  // What follows are a set of flags that indicate useful things for when
  // reading the data in the file.
  bool ConstantTSC : 1;
  bool NonstopTSC : 1;

  // The frequency by which TSC increases per-second.
  alignas(8) uint64_t CycleFrequency = 0;
} __attribute__((packed));

static_assert(sizeof(XRayFileHeader) == 32, "XRayFileHeader != 32 bytes");

enum RecordTypes {
  NORMAL = 0,
};

struct alignas(32) XRayRecord {
  // This is the type of the record being written. We use 16 bits to allow us to
  // treat this as a discriminant, and so that the first 4 bytes get packed
  // properly. See RecordTypes for more supported types.
  uint16_t RecordType = 0;

  // The CPU where the thread is running. We assume number of CPUs <= 256.
  uint8_t CPU = 0;

  // The type of the event. Usually either ENTER = 0 or EXIT = 1.
  uint8_t Type = 0;

  // The function ID for the record.
  int32_t FuncId = 0;

  // Get the full 8 bytes of the TSC when we get the log record.
  uint64_t TSC = 0;

  // The thread ID for the currently running thread.
  uint32_t TId = 0;

  // Use some bytes in the end of the record for buffers.
  char Buffer[4] = {};
} __attribute__((packed));

static_assert(sizeof(XRayRecord) == 32, "XRayRecord != 32 bytes");

} // namespace __xray

#endif // XRAY_XRAY_RECORDS_H
