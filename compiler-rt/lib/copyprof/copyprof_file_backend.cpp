//===-- copyprof_file_backend.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the binary file reporting backend for CopyProf,
/// serializing copy profiling reports into raw indexed profile data files.
///
//===----------------------------------------------------------------------===//

#include "copyprof_file_backend.h"

#include "copyprof_report.h"
#include "copyprof_report_buffer.h"
#include "profile/CopyProfData.inc"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_array_ref.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __copyprof {
namespace {

using ::__sanitizer::ListOfModules;
using ::__sanitizer::LoadedModule;
using ::llvm::copyprof::BuildIdMaxSize;
using ::llvm::copyprof::Header;
using ::llvm::copyprof::RawMagic64;
using ::llvm::copyprof::SegmentEntry;
using AddressRange = ::__sanitizer::LoadedModule::AddressRange;

// Note: manual endianness handling is used here because compiler-rt
// does not have access to LLVM's Support/Endian.h to avoid circular
// dependencies.
// Returns a pointer to the position after which `pod` has been written to
// `buffer`.
template <class T>
char* WriteBytes(const T& pod, char* buffer) {
  const u8* src = reinterpret_cast<const u8*>(&pod);
  for (usize i = 0; i < sizeof(T); ++i)
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // Reverse byte order since reader is little-endian.
    buffer[i] = src[sizeof(T) - 1 - i];
#else
    buffer[i] = src[i];
#endif
  return buffer + sizeof(T);
}

u64 SegmentSizeBytes(ArrayRef<LoadedModule> modules) {
  u64 num_segments_to_record = 0;
  for (const LoadedModule& module : modules) {
    for (const AddressRange& segment : module.ranges()) {
      if (segment.executable)
        ++num_segments_to_record;
    }
  }
  // Count + N*SegmentEntry.
  return sizeof(u64) + sizeof(SegmentEntry) * num_segments_to_record;
}

char* SerializeSegmentsToBuffer(ArrayRef<LoadedModule> modules,
                                u64 expected_num_bytes, char* buffer) {
  char* ptr = buffer;
  ptr += sizeof(u64);  // Reserve space for number of segments
  u64 num_segments_recorded = 0;
  for (const LoadedModule& module : modules) {
    for (const AddressRange& segment : module.ranges()) {
      if (segment.executable) {
        SegmentEntry entry;
        entry.Start = segment.beg;
        entry.End = segment.end;
        entry.Offset = module.base_address();
        uptr uuid_size = module.uuid_size();
        if (uuid_size > BuildIdMaxSize)
          uuid_size = BuildIdMaxSize;
        entry.BuildIdSize = uuid_size;
        internal_memset(entry.BuildId, 0, BuildIdMaxSize);
        internal_memcpy(entry.BuildId, module.uuid(), uuid_size);
        internal_memcpy(ptr, &entry, sizeof(SegmentEntry));
        ptr += sizeof(SegmentEntry);
        ++num_segments_recorded;
      }
    }
  }
  *(reinterpret_cast<u64*>(buffer)) = num_segments_recorded;
  CHECK(expected_num_bytes >= static_cast<u64>(ptr - buffer) &&
        "Expected num bytes != actual bytes written");
  return buffer + expected_num_bytes;
}

u64 StackSizeBytes(uptr num_ids) {
  u64 num_bytes_to_write = sizeof(u64);  // For the count of stacks

  for (u32 id = 1; id <= num_ids; ++id) {
    const StackTrace st = StackDepotGet(id);
    if (st.trace == nullptr || st.size == 0)
      continue;

    // One entry for the id and then one more for the number of stack pcs.
    num_bytes_to_write += 2 * sizeof(u64);
    for (uptr i = 0; i < st.size && st.trace[i] != 0; i++) {
      num_bytes_to_write += sizeof(u64);
    }
  }
  return num_bytes_to_write;
}

void SerializeStackToBuffer(uptr num_ids, u64 expected_num_bytes,
                            char*& buffer) {
  char* ptr = buffer;
  ptr += sizeof(u64);  // Reserve space for count of stacks
  u64 count_stored = 0;

  for (u32 id = 1; id <= num_ids; ++id) {
    const StackTrace st = StackDepotGet(id);
    if (st.trace == nullptr || st.size == 0)
      continue;

    ptr = WriteBytes(static_cast<u64>(id), ptr);
    ptr += sizeof(u64);  // Reserve space for frame count

    u64 frame_count = 0;
    for (uptr i = 0; i < st.size && st.trace[i] != 0; i++) {
      uptr pc = StackTrace::GetPreviousInstructionPc(st.trace[i]);
      ptr = WriteBytes(static_cast<u64>(pc), ptr);
      frame_count++;
    }
    // Store the frame count in the space we reserved earlier.
    *(u64*)(ptr - (frame_count + 1) * sizeof(u64)) = frame_count;
    count_stored++;
  }
  *((u64*)buffer) = count_stored;
  CHECK(expected_num_bytes >= static_cast<u64>(ptr - buffer) &&
        "Expected num bytes != actual bytes written");
}

}  // namespace

FileBackend* FileBackend::Create(const char* path) {
  if (!path || path[0] == '\0') {
    return nullptr;
  }
  InternalScopedString profile_path;
  profile_path.AppendF("%s.%zd.%llu.copyprof", path, internal_getpid(),
                       NanoTime());
  fd_t fd = OpenFile(profile_path.data(), WrOnly);
  if (fd == kInvalidFd) {
    Printf("[copyprof] ERROR: Failed to open profile file: %s.\n",
           profile_path.data());
    return nullptr;
  }
  return new (InternalAlloc(sizeof(FileBackend))) FileBackend(fd);
}

FileBackend::FileBackend(fd_t report_fd) : report_fd_(report_fd) {
  // Write dummy header to reserve space.
  Header header;
  internal_memset(&header, 0, sizeof(header));
  WriteToFile(report_fd_, &header, sizeof(header));
}

void FileBackend::Flush(ReportBuffer* buffer) {
  const char* data = buffer->data();
  usize pos = buffer->pos();
  InternalScopedString output;
  for (usize offset = 0; offset + sizeof(CopyProfReport) <= pos;
       offset += sizeof(CopyProfReport)) {
    output.clear();
    const auto* report = reinterpret_cast<const CopyProfReport*>(data + offset);
    WriteToFile(report_fd_, report, sizeof(CopyProfReport));
    ++num_reports_;
  }
}

void FileBackend::Finalize() {
  uptr num_ids = StackDepotGetStats().n_uniq_ids;
  ListOfModules list;
  list.init();
  ArrayRef<LoadedModule> modules(list.begin(), list.end());
  u64 num_segment_bytes = RoundUpTo(SegmentSizeBytes(modules), 8);
  u64 num_stack_bytes = RoundUpTo(StackSizeBytes(num_ids), 8);
  uptr segment_offset = internal_lseek(report_fd_, 0, 1 /* SEEK_CUR */);
  u64 extra_bytes = num_segment_bytes + num_stack_bytes;
  char* buffer = reinterpret_cast<char*>(InternalAlloc(extra_bytes));
  char* ptr = SerializeSegmentsToBuffer(modules, num_segment_bytes, buffer);
  SerializeStackToBuffer(num_ids, num_stack_bytes, ptr);
  WriteToFile(report_fd_, buffer, extra_bytes);
  InternalFree(buffer);

  // Finally, fix header at the beginning of the file.
  uptr stack_offset = segment_offset + num_segment_bytes;
  Header header;
  header.Magic = RawMagic64;
  header.Version = 1;
  header.ReportOffset = sizeof(Header);
  header.NumReports = num_reports_;
  header.SegmentOffset = segment_offset;
  header.StackOffset = stack_offset;
  internal_lseek(report_fd_, 0, 0 /* SEEK_SET */);
  WriteToFile(report_fd_, &header, sizeof(header));
  CloseFile(report_fd_);
}

}  // namespace __copyprof
