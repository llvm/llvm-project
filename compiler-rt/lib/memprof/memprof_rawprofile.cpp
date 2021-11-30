#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "memprof_meminfoblock.h"
#include "memprof_rawprofile.h"
#include "profile/MemProfData.inc"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stackdepotbase.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_vector.h"

namespace __memprof {
using ::__sanitizer::Vector;
using SegmentEntry = ::llvm::memprof::SegmentEntry;
using Header = ::llvm::memprof::Header;

namespace {
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
      SegmentEntry Entry{};
      Entry.Start = segment.start;
      Entry.End = segment.end;
      Entry.Offset = segment.offset;
      memcpy(Entry.BuildId, segment.uuid, sizeof(segment.uuid));
      memcpy(Ptr, &Entry, sizeof(SegmentEntry));
      Ptr += sizeof(SegmentEntry);
      NumSegmentsRecorded++;
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
// ----------
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
u64 SerializeToRawProfile(MIBMapTy &MIBMap, MemoryMappingLayoutBase &Layout,
                          char *&Buffer) {
  // Each section size is rounded up to 8b since the first entry in each section
  // is a u64 which holds the number of entries in the section by convention.
  const u64 NumSegmentBytes = RoundUpTo(SegmentSizeBytes(Layout), 8);

  Vector<u64> StackIds;
  MIBMap.ForEach(RecordStackId, reinterpret_cast<void *>(&StackIds));
  // The first 8b are for the total number of MIB records. Each MIB record is
  // preceded by a 8b stack id which is associated with stack frames in the next
  // section.
  const u64 NumMIBInfoBytes = RoundUpTo(
      sizeof(u64) + StackIds.Size() * (sizeof(u64) + sizeof(MemInfoBlock)), 8);

  const u64 NumStackBytes = RoundUpTo(StackSizeBytes(StackIds), 8);

  // Ensure that the profile is 8b aligned. We allow for some optional padding
  // at the end so that any subsequent profile serialized to the same file does
  // not incur unaligned accesses.
  const u64 TotalSizeBytes = RoundUpTo(
      sizeof(Header) + NumSegmentBytes + NumStackBytes + NumMIBInfoBytes, 8);

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
