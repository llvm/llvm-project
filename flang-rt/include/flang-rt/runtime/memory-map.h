//===-- include/flang-rt/runtime/memory-map.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Optional copy-out compatibility feature: consult the process memory map to
// recognize copy-out destinations that live in read-only memory and skip the
// write-back (see RTDEF(CopyOutAssign)). A compiler-generated copy-out into
// genuinely read-only storage could only ever rewrite identical bytes or
// fault, so skipping converts the fault into a no-op for programs that
// (invalidly) modified a temporary whose original is not definable.
//
// Modes (FLANG_RT_COPYOUT_READONLY_MODE): 0 = off (default), 1 = trust the
// one-time lazy snapshot (no system calls on the copy-out path), 2 = re-confirm
// each snapshot hit against the current OS state before skipping. Every
// uncertainty - unsupported platform, parse anomaly, allocation failure,
// degenerate descriptor, partial containment - answers "not read-only", i.e.
// the regular copy-out runs. Host-only: all of this is compiled out of device
// paths.

#ifndef FLANG_RT_RUNTIME_MEMORY_MAP_H_
#define FLANG_RT_RUNTIME_MEMORY_MAP_H_

#include "flang/Common/api-attrs.h"
#include <cstddef>
#include <cstdint>

namespace Fortran::runtime {
class Descriptor;

#if !defined(RT_DEVICE_COMPILATION) && !defined(RT_GPU_TARGET)

enum class CopyOutReadOnlyMode : int {
  Off = 0, // feature disabled
  Trust = 1, // snapshot table only; no system calls at copy-out
  Confirm = 2, // snapshot hit re-confirmed against current OS state
};

// Parsed lazily from FLANG_RT_COPYOUT_READONLY_MODE; invalid values are Off.
CopyOutReadOnlyMode GetCopyOutReadOnlyMode();

// True iff the whole data span of 'var' lies within the snapshot's read-only
// regions. Performs no system calls after the one-time lazy snapshot.
bool CopyOutReadOnlyCandidate(const Descriptor &var);

// True iff the whole data span of 'var' is mapped read-only in the *current*
// OS memory map. Performs system calls; used by mode 2 on candidate hits.
bool CopyOutReadOnlyConfirm(const Descriptor &var);

// Diagnostics (FLANG_RT_COPYOUT_READONLY_DIAG=1): first-N notes on stderr and
// a process-lifetime counter. Never allocates and never blocks.
void NoteSkippedCopyOut(const char *sourceFile, int sourceLine);

// Internal pieces exposed for unit testing only.
namespace memmap {
struct Region {
  std::uintptr_t start;
  std::uintptr_t end; // exclusive
};

// Parses a complete /proc/self/maps-format buffer into a malloc'd, coalesced,
// ascending Region array of the read-only entries. fileBackedOnly selects the
// trust-table filter (r, no w, private, inode != 0, no [pseudo] or (deleted)
// paths); otherwise any readable non-writable mapping qualifies. Returns false
// on ANY anomaly (malformed line, out-of-order or overlapping entries,
// overflow, allocation failure) without publishing a partial result.
bool ParseProcMaps(const char *buf, std::size_t len, bool fileBackedOnly,
    Region **out, std::size_t *outCount);

// Windows MEMORY_BASIC_INFORMATION classification, compiled everywhere so it
// is unit-testable on any host. imageOnly selects the trust-table filter
// (MEM_IMAGE regions only). Rejects guard pages, write-copy, and every
// unknown protection combination.
bool ProtectionIsReadOnly(std::uint32_t state, std::uint32_t protect,
    std::uint32_t type, bool imageOnly);

// True iff [lo, hi) is fully contained in one entry of the ascending,
// coalesced region array.
bool SpanIsContained(std::uintptr_t lo, std::uintptr_t hi,
    const Region *regions, std::size_t count);

// Computes the byte span touched through 'var' (stride-sign-aware, overflow
// checked). False for degenerate descriptors (unallocated, zero extent,
// zero-length elements) and on arithmetic overflow.
bool ComputeDataSpan(
    const Descriptor &var, std::uintptr_t &lo, std::uintptr_t &hi);
} // namespace memmap

#endif // !RT_DEVICE_COMPILATION && !RT_GPU_TARGET

} // namespace Fortran::runtime
#endif // FLANG_RT_RUNTIME_MEMORY_MAP_H_
