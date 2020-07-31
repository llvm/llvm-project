//===-- crash_handler_interface.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/common.h"
#include "gwp_asan/stack_trace_compressor.h"

#include <assert.h>
#include <string.h>

using AllocationMetadata = gwp_asan::AllocationMetadata;
using Error = gwp_asan::Error;

#ifdef __cplusplus
extern "C" {
#endif

bool __gwp_asan_error_is_mine(const gwp_asan::AllocatorState *State,
                              uintptr_t ErrorPtr) {
  assert(State && "State should not be nullptr.");
  if (State->FailureType != Error::UNKNOWN && State->FailureAddress != 0)
    return true;

  return ErrorPtr < State->GuardedPagePoolEnd &&
         State->GuardedPagePool <= ErrorPtr;
}

uintptr_t
__gwp_asan_get_internal_crash_address(const gwp_asan::AllocatorState *State) {
  return State->FailureAddress;
}

static const AllocationMetadata *
addrToMetadata(const gwp_asan::AllocatorState *State,
               const AllocationMetadata *Metadata, uintptr_t Ptr) {
  // Note - Similar implementation in guarded_pool_allocator.cpp.
  return &Metadata[State->getNearestSlot(Ptr)];
}

gwp_asan::Error
__gwp_asan_diagnose_error(const gwp_asan::AllocatorState *State,
                          const gwp_asan::AllocationMetadata *Metadata,
                          uintptr_t ErrorPtr) {
  if (!__gwp_asan_error_is_mine(State, ErrorPtr))
    return Error::UNKNOWN;

  if (State->FailureType != Error::UNKNOWN)
    return State->FailureType;

  // Let's try and figure out what the source of this error is.
  if (State->isGuardPage(ErrorPtr)) {
    size_t Slot = State->getNearestSlot(ErrorPtr);
    const AllocationMetadata *SlotMeta =
        addrToMetadata(State, Metadata, State->slotToAddr(Slot));

    // Ensure that this slot was allocated once upon a time.
    if (!SlotMeta->Addr)
      return Error::UNKNOWN;

    if (SlotMeta->Addr < ErrorPtr)
      return Error::BUFFER_OVERFLOW;
    return Error::BUFFER_UNDERFLOW;
  }

  // Access wasn't a guard page, check for use-after-free.
  const AllocationMetadata *SlotMeta =
      addrToMetadata(State, Metadata, ErrorPtr);
  if (SlotMeta->IsDeallocated) {
    return Error::USE_AFTER_FREE;
  }

  // If we have reached here, the error is still unknown.
  return Error::UNKNOWN;
}

const gwp_asan::AllocationMetadata *
__gwp_asan_get_metadata(const gwp_asan::AllocatorState *State,
                        const gwp_asan::AllocationMetadata *Metadata,
                        uintptr_t ErrorPtr) {
  if (!__gwp_asan_error_is_mine(State, ErrorPtr))
    return nullptr;

  if (ErrorPtr >= State->GuardedPagePoolEnd ||
      State->GuardedPagePool > ErrorPtr)
    return nullptr;

  const AllocationMetadata *Meta = addrToMetadata(State, Metadata, ErrorPtr);
  if (Meta->Addr == 0)
    return nullptr;

  return Meta;
}

uintptr_t __gwp_asan_get_allocation_address(
    const gwp_asan::AllocationMetadata *AllocationMeta) {
  return AllocationMeta->Addr;
}

size_t __gwp_asan_get_allocation_size(
    const gwp_asan::AllocationMetadata *AllocationMeta) {
  return AllocationMeta->Size;
}

uint64_t __gwp_asan_get_allocation_thread_id(
    const gwp_asan::AllocationMetadata *AllocationMeta) {
  return AllocationMeta->AllocationTrace.ThreadID;
}

size_t __gwp_asan_get_allocation_trace(
    const gwp_asan::AllocationMetadata *AllocationMeta, uintptr_t *Buffer,
    size_t BufferLen) {
  uintptr_t UncompressedBuffer[AllocationMetadata::kMaxTraceLengthToCollect];
  size_t UnpackedLength = gwp_asan::compression::unpack(
      AllocationMeta->AllocationTrace.CompressedTrace,
      AllocationMeta->AllocationTrace.TraceSize, UncompressedBuffer,
      AllocationMetadata::kMaxTraceLengthToCollect);
  if (UnpackedLength < BufferLen)
    BufferLen = UnpackedLength;
  memcpy(Buffer, UncompressedBuffer, BufferLen * sizeof(*Buffer));
  return UnpackedLength;
}

bool __gwp_asan_is_deallocated(
    const gwp_asan::AllocationMetadata *AllocationMeta) {
  return AllocationMeta->IsDeallocated;
}

uint64_t __gwp_asan_get_deallocation_thread_id(
    const gwp_asan::AllocationMetadata *AllocationMeta) {
  return AllocationMeta->DeallocationTrace.ThreadID;
}

size_t __gwp_asan_get_deallocation_trace(
    const gwp_asan::AllocationMetadata *AllocationMeta, uintptr_t *Buffer,
    size_t BufferLen) {
  uintptr_t UncompressedBuffer[AllocationMetadata::kMaxTraceLengthToCollect];
  size_t UnpackedLength = gwp_asan::compression::unpack(
      AllocationMeta->DeallocationTrace.CompressedTrace,
      AllocationMeta->DeallocationTrace.TraceSize, UncompressedBuffer,
      AllocationMetadata::kMaxTraceLengthToCollect);
  if (UnpackedLength < BufferLen)
    BufferLen = UnpackedLength;
  memcpy(Buffer, UncompressedBuffer, BufferLen * sizeof(*Buffer));
  return UnpackedLength;
}

#ifdef __cplusplus
} // extern "C"
#endif
