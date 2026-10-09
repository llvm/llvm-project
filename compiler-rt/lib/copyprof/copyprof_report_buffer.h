//===-- copyprof_report_buffer.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the report buffer, a fixed-size bump allocator in which
/// threads store their CopyProf reports until they are flushed.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_REPORT_BUFFER_H_
#define COPYPROF_REPORT_BUFFER_H_

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __copyprof {

// A fixed-size buffer offering a simple bump-allocation mechanism.
// Used to store `CopyProfReport`s. Thread-compatible.
struct ReportBuffer final {
 public:
  // The size in bytes of the buffer usable for reports.
  // The buffer starts after the `ReportBuffer` instance in memory.
  explicit ReportBuffer(usize buffer_size);
  ReportBuffer(const ReportBuffer&) = delete;
  ReportBuffer& operator=(const ReportBuffer&) = delete;

  // Returns a pointer to the buffer of the given size, or `nullptr` if
  // `num_bytes` bytes exceeds the buffer's capacity.
  void* TryAllocate(usize num_bytes);

  // Must be called after the buffer has been consumed by the sink and before it
  // can be acquired again by a thread.
  void Reset();

  // `ReportBuffer` instances are part of an `IntrusiveList` so need a `next`
  // pointer.
  ReportBuffer* next = nullptr;

  // Pointer to the beginning of the report buffer.
  char* data();
  const char* data() const;

  // Current offset (in bytes) into the report buffer, i.e., the number of
  // bytes that have been allocated so far.
  usize pos() const;

 private:
  const usize buffer_size_;
  usize pos_ = 0;
};

}  // namespace __copyprof

#endif  // COPYPROF_REPORT_BUFFER_H_
