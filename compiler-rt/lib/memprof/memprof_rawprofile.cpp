#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "memprof_rawprofile.h"
#include "profile/MemProfData.inc"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_array_ref.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stackdepotbase.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_vector.h"

namespace __memprof {
using ::__sanitizer::Vector;
using ::llvm::memprof::MemInfoBlock;
using SegmentEntry = ::llvm::memprof::SegmentEntry;
using Header = ::llvm::memprof::Header;

namespace {
template <class T> char *WriteBytes(const T &Pod, char *Buffer) {
  *(T *)Buffer = Pod;
  return Buffer + sizeof(T);
}

void RecordStackId(const uptr Key, UNUSED LockedMemInfoBlock *const &MIB,
                   void *Arg) {
  // No need to touch the MIB value here since we are only recording the key.
  auto *StackIds = reinterpret_cast<Vector<u64> *>(Arg);
  StackIds->PushBack(Key);
}
} // namespace

u64 SegmentSizeBytes(ArrayRef<LoadedModule> Modules) {
  u64 NumSegmentsToRecord = 0;
  for (const auto &Module : Modules) {
    for (const auto &Segment : Module.ranges()) {
      if (Segment.executable)
        NumSegmentsToRecord++;
    }
  }

  return sizeof(u64) // A header which stores the number of records.
         + sizeof(SegmentEntry) * NumSegmentsToRecord;
}

// The segment section uses the following format:
// ---------- Segment Info
// Num Entries
// ---------- Segment Entry
// Start
// End
// Offset
// UuidSize
// Uuid 32B
// ----------
// ...
void SerializeSegmentsToBuffer(ArrayRef<LoadedModule> Modules,
                               const u64 ExpectedNumBytes, char *&Buffer) {
  char *Ptr = Buffer;
  // Reserve space for the final count.
  Ptr += sizeof(u64);

  u64 NumSegmentsRecorded = 0;

  for (const auto &Module : Modules) {
    for (const auto &Segment : Module.ranges()) {
      if (Segment.executable) {
        SegmentEntry Entry(Segment.beg, Segment.end, Module.base_address());
        CHECK(Module.uuid_size() <= MEMPROF_BUILDID_MAX_SIZE);
        Entry.BuildIdSize = Module.uuid_size();
        memcpy(Entry.BuildId, Module.uuid(), Module.uuid_size());
        memcpy(Ptr, &Entry, sizeof(SegmentEntry));
        Ptr += sizeof(SegmentEntry);
        NumSegmentsRecorded++;
      }
    }
  }
  // Store the number of segments we recorded in the space we reserved.
  *((u64 *)Buffer) = NumSegmentsRecorded;
  CHECK(ExpectedNumBytes >= static_cast<u64>(Ptr - Buffer) &&
        "Expected num bytes != actual bytes written");
}

u64 StackSizeBytes(const Vector<u64> &StackIds) {
  u64 NumBytesToWrite = sizeof(u64);

  const u64 NumIds = StackIds.Size();
  for (unsigned k = 0; k < NumIds; ++k) {
    const u64 Id = StackIds[k];
    // One entry for the id and then one more for the number of stack pcs.
    NumBytesToWrite += 2 * sizeof(u64);
    const StackTrace St = StackDepotGet(Id);

    CHECK(St.trace != nullptr && St.size > 0 && "Empty stack trace");
    for (uptr i = 0; i < St.size && St.trace[i] != 0; i++) {
      NumBytesToWrite += sizeof(u64);
    }
  }
  return NumBytesToWrite;
}

// The stack info section uses the following format:
//
// ---------- Stack Info
// Num Entries
// ---------- Stack Entry
// Num Stacks
// PC1
// PC2
// ...
// ----------
void SerializeStackToBuffer(const Vector<u64> &StackIds,
                            const u64 ExpectedNumBytes, char *&Buffer) {
  const u64 NumIds = StackIds.Size();
  char *Ptr = Buffer;
  Ptr = WriteBytes(static_cast<u64>(NumIds), Ptr);

  for (unsigned k = 0; k < NumIds; ++k) {
    const u64 Id = StackIds[k];
    Ptr = WriteBytes(Id, Ptr);
    Ptr += sizeof(u64); // Bump it by u64, we will fill this in later.
    u64 Count = 0;
    const StackTrace St = StackDepotGet(Id);
    for (uptr i = 0; i < St.size && St.trace[i] != 0; i++) {
      // PCs in stack traces are actually the return addresses, that is,
      // addresses of the next instructions after the call.
      uptr pc = StackTrace::GetPreviousInstructionPc(St.trace[i]);
      Ptr = WriteBytes(static_cast<u64>(pc), Ptr);
      ++Count;
    }
    // Store the count in the space we reserved earlier.
    *(u64 *)(Ptr - (Count + 1) * sizeof(u64)) = Count;
  }

  CHECK(ExpectedNumBytes >= static_cast<u64>(Ptr - Buffer) &&
        "Expected num bytes != actual bytes written");
}

// The MIB section has the following format:
// ---------- MIB Info
// Num Entries
// ---------- MIB Entry 0
// Alloc Count
// ...
//       ---- AccessHistogram Entry 0
//            ...
//       ---- AccessHistogram Entry AccessHistogramSize - 1
// ---------- MIB Entry 1
// Alloc Count
// ...
//       ---- AccessHistogram Entry 0
//            ...
//       ---- AccessHistogram Entry AccessHistogramSize - 1
// ----------
void SerializeMIBInfoToBuffer(MIBMapTy &MIBMap, const Vector<u64> &StackIds,
                              const u64 ExpectedNumBytes, char *&Buffer) {
  char *Ptr = Buffer;
  const u64 NumEntries = StackIds.Size();
  Ptr = WriteBytes(NumEntries, Ptr);
  for (u64 i = 0; i < NumEntries; i++) {
    const u64 Key = StackIds[i];
    MIBMapTy::Handle h(&MIBMap, Key, /*remove=*/true, /*create=*/false);
    CHECK(h.exists());
    Ptr = WriteBytes(Key, Ptr);
    // FIXME: We unnecessarily serialize the AccessHistogram pointer. Adding a
    // serialization schema will fix this issue. See also FIXME in
    // deserialization.
    Ptr = WriteBytes((*h)->mib, Ptr);
    for (u64 j = 0; j < (*h)->mib.AccessHistogramSize; ++j) {
      u64 HistogramEntry = ((u64 *)((*h)->mib.AccessHistogram))[j];
      Ptr = WriteBytes(HistogramEntry, Ptr);
    }
    if ((*h)->mib.AccessHistogramSize > 0) {
      InternalFree((void *)((*h)->mib.AccessHistogram));
    }
  }
  CHECK(ExpectedNumBytes >= static_cast<u64>(Ptr - Buffer) &&
        "Expected num bytes != actual bytes written");
}

// Format
// ---------- Header
// Magic
// Version
// Total Size
// Segment Offset
// MIB Info Offset
// Stack Offset
// ---------- Segment Info
// Num Entries
// ---------- Segment Entry
// Start
// End
// Offset
// BuildID 32B
// ----------
// ...
// ----------
// Optional Padding Bytes
// ---------- MIB Info
// Num Entries
// ---------- MIB Entry
// Alloc Count
// ...
//       ---- AccessHistogram Entry 0
//            ...
//       ---- AccessHistogram Entry AccessHistogramSize - 1
// ---------- MIB Entry 1
// Alloc Count
// ...
//       ---- AccessHistogram Entry 0
//            ...
//       ---- AccessHistogram Entry AccessHistogramSize - 1
// Optional Padding Bytes
// ---------- Stack Info
// Num Entries
// ---------- Stack Entry
// Num Stacks
// PC1
// PC2
// ...
// ----------
// Optional Padding Bytes
// ...
u64 SerializeToRawProfile(MIBMapTy &MIBMap, ArrayRef<LoadedModule> Modules,
                          char *&Buffer) {
  // Each section size is rounded up to 8b since the first entry in each section
  // is a u64 which holds the number of entries in the section by convention.
  const u64 NumSegmentBytes = RoundUpTo(SegmentSizeBytes(Modules), 8);

  Vector<u64> StackIds;
  MIBMap.ForEach(RecordStackId, reinterpret_cast<void *>(&StackIds));
  // The first 8b are for the total number of MIB records. Each MIB record is
  // preceded by a 8b stack id which is associated with stack frames in the next
  // section.
  const u64 NumMIBInfoBytes = RoundUpTo(
      sizeof(u64) + StackIds.Size() * (sizeof(u64) + sizeof(MemInfoBlock)), 8);

  // Get Number of AccessHistogram entries in total
  u64 TotalAccessHistogramEntries = 0;
  MIBMap.ForEach(
      [](const uptr Key, UNUSED LockedMemInfoBlock *const &MIB, void *Arg) {
        u64 *TotalAccessHistogramEntries = (u64 *)Arg;
        *TotalAccessHistogramEntries += MIB->mib.AccessHistogramSize;
      },
      reinterpret_cast<void *>(&TotalAccessHistogramEntries));
  const u64 NumHistogramBytes =
      RoundUpTo(TotalAccessHistogramEntries * sizeof(uint64_t), 8);

  const u64 NumStackBytes = RoundUpTo(StackSizeBytes(StackIds), 8);

  // Ensure that the profile is 8b aligned. We allow for some optional padding
  // at the end so that any subsequent profile serialized to the same file does
  // not incur unaligned accesses.
  const u64 TotalSizeBytes =
      RoundUpTo(sizeof(Header) + NumSegmentBytes + NumStackBytes +
                    NumMIBInfoBytes + NumHistogramBytes,
                8);

  // Allocate the memory for the entire buffer incl. info blocks.
  Buffer = (char *)InternalAlloc(TotalSizeBytes);
  char *Ptr = Buffer;

  Header header{MEMPROF_RAW_MAGIC_64,
                MEMPROF_RAW_VERSION,
                static_cast<u64>(TotalSizeBytes),
                sizeof(Header),
                sizeof(Header) + NumSegmentBytes,
                sizeof(Header) + NumSegmentBytes + NumMIBInfoBytes +
                    NumHistogramBytes};
  Ptr = WriteBytes(header, Ptr);

  SerializeSegmentsToBuffer(Modules, NumSegmentBytes, Ptr);
  Ptr += NumSegmentBytes;

  SerializeMIBInfoToBuffer(MIBMap, StackIds,
                           NumMIBInfoBytes + NumHistogramBytes, Ptr);
  Ptr += NumMIBInfoBytes + NumHistogramBytes;

  SerializeStackToBuffer(StackIds, NumStackBytes, Ptr);

  return TotalSizeBytes;
}

/*
  The format of the binary access profile:
  // header
  BinaryAccessHeader header;
  // segment info
  SegmentEntry entry1;
  SegmentEntry entry2;
  ...
  // memblock addresses
  u64 MemBlockAddress1;
  u64 MemBlockAddress2;
  ...
  // end

BinaryAccessHeader is defined in MemProfBinaryAccessData.inc
PACKED(struct BinaryAccessHeader {
  uint64_t Magic;
  uint64_t Version;
  uint64_t TotalSize;
  uint64_t SegmentOffset;
  uint64_t NumSegments;
  uint64_t MemAddressOffset;
  uint64_t NumMemBlockAddresses;
});
SegmentEntry is defined in MemProfData.inc
  struct SegmentEntry {
  uint64_t Start; // segment start address
  uint64_t End;   // segment end address
  uint64_t Offset;  // binary offset at runtime
  uint64_t BuildIdSize;
  uint8_t BuildId[MEMPROF_BUILDID_MAX_SIZE] = {0};
#define MEMPROF_BUILDID_MAX_SIZE 32ULL
*/

using BinaryAccessHeader = ::llvm::memprof::BinaryAccessHeader;

u64 SerializeBinaryAccesses(ArrayRef<LoadedModule> Modules,
                            char *&BufferStart) {
  // Serialize the contents to a raw profile.
  Vector<SegmentEntry> Entries;
  Vector<u64> MemBlockAddresses;
  for (const auto &Module : Modules) {
    // TODO: is there a better way to filter the binaries we care?
    if (strstr(Module.full_name(), ".app"))
      for (const auto &Segment : Module.ranges()) {
        // collect segment info
        SegmentEntry Entry(Segment.beg, Segment.end, Module.base_address());
        Entry.BuildIdSize = Module.uuid_size();
        memcpy(Entry.BuildId, Module.uuid(), Module.uuid_size());
        Entries.PushBack(Entry);
        // collect memblock addresses whose access > 0 for each segment
        for (uptr t = Entry.Start & SHADOW_MASK; t < Entry.End;
             t += MEM_GRANULARITY)
          if (GetShadowCount(t, MEM_GRANULARITY - 4) > 0)
            MemBlockAddresses.PushBack(t);
      }
  }
  // Allocate the memory for the entire buffer incl binaries segment info
  // and memblock addresses
  u64 NumSegmentsToRecord = Entries.Size();
  u64 NumMemBlockAddressesToRecord = MemBlockAddresses.Size();
  u64 NumSegmentsToRecordBytes =
      RoundUpTo(sizeof(SegmentEntry) * NumSegmentsToRecord, 8);
  u64 NumMemBlockAddressesToRecordBytes =
      RoundUpTo(NumMemBlockAddressesToRecord * sizeof(u64), 8);
  u64 NumHeaderBytes = RoundUpTo(sizeof(BinaryAccessHeader), 8);
  u64 TotalBytes = NumHeaderBytes + NumSegmentsToRecordBytes +
                   NumMemBlockAddressesToRecordBytes;
  BufferStart = (char *)InternalAlloc(TotalBytes);
  char *Buffer = BufferStart;

  BinaryAccessHeader header{MEMPROF_BINARY_ACCESS_RAW_MAGIC_64,
                            MEMPROF_BINARY_ACCESS_RAW_VERSION,
                            TotalBytes,
                            NumHeaderBytes,
                            NumSegmentsToRecord,
                            NumHeaderBytes + NumSegmentsToRecordBytes,
                            NumMemBlockAddressesToRecord};
  memcpy(Buffer, &header, sizeof(BinaryAccessHeader));
  Buffer += NumHeaderBytes;
  for (unsigned k = 0; k < NumSegmentsToRecord; k++) {
    memcpy(Buffer, &Entries[k], sizeof(SegmentEntry));
    Buffer += sizeof(SegmentEntry);
  }
  for (unsigned k = 0; k < NumMemBlockAddressesToRecord; k++) {
    *(uptr *)Buffer = MemBlockAddresses[k];
    Buffer += sizeof(uptr);
  }
  u64 BytesSerialized = Buffer - BufferStart;

  fprintf(
      stderr,
      "[MemProf] NumSegmentsToRecord: %d, NumSegmentsToRecordBytes: %d, "
      "NumMemBlockAddressesToRecord: %d, NumMemBlockAddressesToRecordBytes: "
      "%d, BytesSerialized: %d, Buffer: %p\n",
      NumSegmentsToRecord, NumSegmentsToRecordBytes,
      NumMemBlockAddressesToRecord, NumMemBlockAddressesToRecordBytes,
      BytesSerialized, Buffer);

  return BytesSerialized;
}

void DumpBinaryAccesses() {
  __sanitizer::ListOfModules List;
  List.init();
  ArrayRef<LoadedModule> Modules(List.begin(), List.end());
  for (const auto &Module : Modules) {
    // TODO: is there a better way to filter the binaries we care?
    if (strstr(Module.full_name(), ".app")) {
      for (const auto &Segment : Module.ranges()) {
        SegmentEntry Entry(Segment.beg, Segment.end, Module.base_address());
        Printf("\n[MemProf] BuildId: ");
        for (size_t I = 0; I < Module.uuid_size(); I++)
          Printf("%02x", Module.uuid()[I]);
        Printf("\n");
        Printf("[MemProf] ExecutableName: %s\n", Module.full_name());
        Printf("[MemProf] Start: 0x%zx\n", Entry.Start);

        InternalScopedString shadows;
        for (auto t = Entry.Start & SHADOW_MASK; t < Entry.End;
             t += MEM_GRANULARITY) {
          // should not be 64, as it will include the next shadow memory
          u64 c = GetShadowCount(t, MEM_GRANULARITY - 4);
          if (c > 0)
            shadows.AppendF("[MemProf] Shadow: 0x%zx %d\n", t, c);
        }
        Printf("%s", shadows.data());
        Printf("[MemProf] End: 0x%zx\n", Entry.End);
        Printf("[MemProf] Offset: 0x%zx\n", Entry.Offset);
      }
    }
  }
}
} // namespace __memprof
