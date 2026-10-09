//===-- copyprof_stdout_backend.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the stdout reporting backend interface, providing console
/// logging capabilities for CopyProf reports.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_STDOUT_BACKEND_H_
#define COPYPROF_STDOUT_BACKEND_H_

#include "copyprof_report_backend.h"
#include "copyprof_report_buffer.h"

namespace __copyprof {

// Writes CopyProf reports to stdout.
class StdoutBackend : public ReportBackend {
 protected:
  ~StdoutBackend() = default;

 public:
  static StdoutBackend* Create();
  void Flush(ReportBuffer* buffer) override;
  void Finalize() override;
};

}  // namespace __copyprof

#endif  // COPYPROF_STDOUT_BACKEND_H_
