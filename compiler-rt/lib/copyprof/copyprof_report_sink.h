//===-- copyprof_report_sink.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the background report sink interface, coordinating
/// asynchronous report processing across reporting backends.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_REPORT_SINK_H_
#define COPYPROF_REPORT_SINK_H_

#include "copyprof_report_backend.h"
#include "copyprof_report_buffer.h"

namespace __copyprof {

// Obtains a new memory buffer, or returns `nullptr` if background flushing is
// currently disabled. Thread-safe.
ReportBuffer* AcquireReportBuffer();
// Returns a `ReportBuffer` to the sink so it can be flushed eventually. If
// background flushing is disabled, the buffer's reports are dropped and the
// buffer is reused. Does nothing if `buffer` is `nullptr`. Thread-safe.
void ReturnReportBuffer(ReportBuffer* buffer);

// Must be called by the main thread at initialization time to enable flushing
// of reports by a background thread. Must be preceded by a call to
// `DisableBackgroundFlushing` before it can be called again.
// Takes ownership of the given pointer. The memory is freed in
// `DisableBackgroundFlushing`.
void EnableBackgroundFlushing(ReportBackend* backend);

// Must be called by the main thread typically at program exit, but can be
// safely called arbitrarily often. All report buffers that have previously been
// returned to the sink are flushed (if any). Any report buffers still owned by
// threads (that are still alive and did not return their buffers) are not
// flushed, and such reports will be lost. Afterwards, `AcquireReportBuffer`
// returns `nullptr` and returned buffers are reused without being flushed, so
// threads that keep running drop any further reports. Also closes the report
// output file if one was opened.
void DisableBackgroundFlushing();

}  // namespace __copyprof

#endif  // COPYPROF_REPORT_SINK_H_
