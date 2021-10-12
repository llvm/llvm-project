#include "memprof_rawprofile.h"
#include "memprof_meminfoblock.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stackdepotbase.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_vector.h"

#include <stdlib.h>
#include <string.h>

namespace __memprof {
using ::__sanitizer::Vector;

namespace {
typedef struct __attribute__((__packed__)) {
  u64 start;
  u64 end;
  u64 offset;
  u8 buildId[32];
} SegmentEntry;

typedef struct __attribute__((__packed__)) {
  u64 magic;
  u64 version;
  u64 total_size;
  u64 segment_offset;
  u64 mib_offset;
  u64 stack_offset;
} Header;

template <class T> char *WriteBytes(T Pod, char *&Buffer) {
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

u64 SegmentSizeBytes(MemoryMappingLayoutBase &Layout) {
  u64 NumSegmentsToRecord = 0;
  MemoryMappedSegment segment;
  for (Layout.Reset(); Layout.Next(&segment);)
    if (segment.IsReadable() && segment.IsExecutable())
      NumSegmentsToRecord++;

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
// BuildID 32B
// ----------
// ...
void SerializeSegmentsToBuffer(MemoryMappingLayoutBase &Layout,
                               const u64 ExpectedNumBytes, char *&Buffer) {
  char *Ptr = Buffer;
  // Reserve space for the final count.
  Ptr += sizeof(u64);

  u64 NumSegmentsRecorded = 0;
  MemoryMappedSegment segment;

  for (Layout.Reset(); Layout.Next(&segment);) {
    if (segment.IsReadable() && segment.IsExecutable()) {
      SegmentEntry entry{};
      entry.start = segment.start;
      entry.end = segment.end;
      entry.offset = segment.offset;
      memcpy(entry.buildId, segment.uuid, sizeof(segment.uuid));
      memcpy(Ptr, &entry, sizeof(SegmentEntry));
      Ptr += sizeof(SegmentEntry);
      NumSegmentsRecorded++;
    }
  }

  // Store the number of segments we recorded in the space we reserved.
  *((u64 *)Buffer) = NumSegmentsRecorded;
  CHECK(ExpectedNumBytes == static_cast<u64>(Ptr - Buffer) &&
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

  CHECK(ExpectedNumBytes == static_cast<u64>(Ptr - Buffer) &&
        "Expected num bytes != actual bytes written");
}

// The MIB section has the following format:
// ---------- MIB Info
// Num Entries
// ---------- MIB Entry 0
// Alloc Count
// ...
// ---------- MIB Entry 1
// Alloc Count
// ...
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
    Ptr = WriteBytes((*h)->mib, Ptr);
  }

  CHECK(ExpectedNumBytes == static_cast<u64>(Ptr - Buffer) &&
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
// ---------- MIB Info
// Num Entries
// ---------- MIB Entry
// Alloc Count
// ...
// ---------- Stack Info
// Num Entries
// ---------- Stack Entry
// Num Stacks
// PC1
// PC2
// ...
// ----------
// ...
u64 SerializeToRawProfile(MIBMapTy &MIBMap, MemoryMappingLayoutBase &Layout,
                          char *&Buffer) {
  const u64 NumSegmentBytes = SegmentSizeBytes(Layout);

  Vector<u64> StackIds;
  MIBMap.ForEach(RecordStackId, reinterpret_cast<void *>(&StackIds));
  // The first 8b are for the total number of MIB records. Each MIB record is
  // preceded by a 8b stack id which is associated with stack frames in the next
  // section.
  const u64 NumMIBInfoBytes =
      sizeof(u64) + StackIds.Size() * (sizeof(u64) + sizeof(MemInfoBlock));

  const u64 NumStackBytes = StackSizeBytes(StackIds);

  const u64 TotalSizeBytes =
      sizeof(Header) + NumSegmentBytes + NumStackBytes + NumMIBInfoBytes;

  // Allocate the memory for the entire buffer incl. info blocks.
  Buffer = (char *)InternalAlloc(TotalSizeBytes);
  char *Ptr = Buffer;

  Header header{MEMPROF_RAW_MAGIC_64,
                MEMPROF_RAW_VERSION,
                static_cast<u64>(TotalSizeBytes),
                sizeof(Header),
                sizeof(Header) + NumSegmentBytes,
                sizeof(Header) + NumSegmentBytes + NumMIBInfoBytes};
  Ptr = WriteBytes(header, Ptr);

  SerializeSegmentsToBuffer(Layout, NumSegmentBytes, Ptr);
  Ptr += NumSegmentBytes;

  SerializeMIBInfoToBuffer(MIBMap, StackIds, NumMIBInfoBytes, Ptr);
  Ptr += NumMIBInfoBytes;

  SerializeStackToBuffer(StackIds, NumStackBytes, Ptr);

  return TotalSizeBytes;
}

} // namespace __memprof
