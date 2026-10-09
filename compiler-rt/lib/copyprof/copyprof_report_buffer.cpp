//===-- copyprof_report_buffer.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the lock-free report buffer for CopyProf, temporarily
/// storing profiling reports in memory before background flushing.
///
//===----------------------------------------------------------------------===//

#include "copyprof_report_buffer.h"

#include "sanitizer_common/sanitizer_common.h"

namespace __copyprof {

ReportBuffer::ReportBuffer(usize buffer_size)
    : buffer_size_(buffer_size), pos_(0) {}

void* ReportBuffer::TryAllocate(usize num_bytes) {
  if (pos_ + num_bytes > buffer_size_) {
    return nullptr;
  }
  pos_ += num_bytes;
  return data() + pos_ - num_bytes;
}

void ReportBuffer::Reset() { pos_ = 0; }

char* ReportBuffer::data() {
  // `ReportBuffer` instances are constructed at the beginning of memory block
  // and the user usable buffer starts where the object ends.
  return reinterpret_cast<char*>(&this[1]);
}

const char* ReportBuffer::data() const {
  return const_cast<ReportBuffer*>(this)->data();
}

usize ReportBuffer::pos() const { return pos_; }

}  // namespace __copyprof
