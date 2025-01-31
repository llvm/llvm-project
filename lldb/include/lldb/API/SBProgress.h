//===-- SBProgress.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBPROGRESS_H
#define LLDB_API_SBPROGRESS_H

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDefines.h"

namespace lldb {

/// A Progress indicator helper class.
///
/// Any potentially long running sections of code in LLDB should report
/// progress so that clients are aware of delays that might appear during
/// debugging. Delays commonly include indexing debug information, parsing
/// symbol tables for object files, downloading symbols from remote
/// repositories, and many more things.
///
/// The Progress class helps make sure that progress is correctly reported
/// and will always send an initial progress update, updates when
/// Progress::Increment() is called, and also will make sure that a progress
/// completed update is reported even if the user doesn't explicitly cause one
/// to be sent.
class LLDB_API SBProgress {
public:
  /// Construct a progress object with a title, details and a given debugger.
  /// \param title
  ///   The title of the progress object.
  /// \param details
  ///   The details of the progress object.
  /// \param debugger
  ///   The debugger for this progress object to report to.
  SBProgress(const char *title, const char *details, SBDebugger &debugger);

  /// Construct a progress object with a title, details, the total units of work
  /// to be done, and a given debugger.
  /// \param title
  ///   The title of the progress object.
  /// \param details
  ///   The details of the progress object.
  /// \param total_units
  ///   The total number of units of work to be done.
  /// \param debugger
  ///   The debugger for this progress object to report to.
  SBProgress(const char *title, const char *details, uint64_t total_units,
             SBDebugger &debugger);

#ifndef SWIG
  SBProgress(SBProgress &&rhs);
#endif

  ~SBProgress();

  void Increment(uint64_t amount, const char *description = nullptr);

protected:
  lldb_private::Progress &ref() const;

private:
  SBProgress(const SBProgress &rhs) = delete;
  const SBProgress &operator=(const SBProgress &rhs) = delete;

  std::unique_ptr<lldb_private::Progress> m_opaque_up;
}; // SBProgress
} // namespace lldb

#endif // LLDB_API_SBPROGRESS_H
