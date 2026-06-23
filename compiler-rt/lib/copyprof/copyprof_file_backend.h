//===-- copyprof_file_backend.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the binary file reporting backend interface, enabling
/// CopyProf reports to be saved to disk for offline analysis and tooling.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_FILE_BACKEND_H_
#define COPYPROF_FILE_BACKEND_H_

#include "copyprof_report_backend.h"
#include "copyprof_report_buffer.h"

namespace __copyprof {

// Writes report buffers to a file.
class FileBackend : public ReportBackend {
 protected:
  ~FileBackend() = default;

 public:
  static FileBackend* Create(const char* path);
  void Flush(ReportBuffer* buffer) override;
  void Finalize() override;

 private:
  explicit FileBackend(fd_t report_fd);

  fd_t report_fd_;
  u64 num_reports_ = 0;
};

}  // namespace __copyprof

#endif  // COPYPROF_FILE_BACKEND_H_
