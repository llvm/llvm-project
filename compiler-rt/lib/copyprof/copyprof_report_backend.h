//===-- copyprof_report_backend.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the abstract report backend interface for CopyProf,
/// enabling custom destinations for copy profiling reports.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_REPORT_BACKEND_H_
#define COPYPROF_REPORT_BACKEND_H_

#include "copyprof_report_buffer.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __copyprof {

class ReportBackend {
 protected:
  // Subtle: `ReportBackend` instances are never deleted, but the ABI requires
  // that a class w/ a virtual d'tor must emit a reference to a deleting
  // destructor into its vtable. Thus, the d'tor is not delcared as `virtual`.
  // Similarly, a class cannot have pure virtual methods as that would pull in a
  // dependency on `__cxa_pure_virtual`.
  ~ReportBackend() = default;

 public:
  // Called each time a report buffer should be flushed to the backend.
  virtual void Flush(ReportBuffer* buffer) { UNIMPLEMENTED(); };
  // Called exactly once before the sink shuts down and reporting is disabled.
  virtual void Finalize() { UNIMPLEMENTED(); };
};

}  // namespace __copyprof

#endif  // COPYPROF_REPORT_BACKEND_H_
