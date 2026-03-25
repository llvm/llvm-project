//===-- TraceExporter.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TRACEEXPORTER_H
#define LLDB_TARGET_TRACEEXPORTER_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-forward.h"
#include "llvm/Support/Error.h"

namespace lldb_private {

/// \class TraceExporter TraceExporter.h "lldb/Target/TraceExporter.h"
/// A plug-in interface definition class for trace exporters.
///
/// Trace exporter plug-ins operate on traces, converting the trace data
/// provided by an \a lldb_private::TraceCursor into a different format that can
/// be digested by other tools, e.g. Chrome Trace Event Profiler.
///
/// These plugins enable interoperability by exporting processor trace data
/// collected by LLDB into formats that can be consumed by external analysis
/// tools, visualization software, or other debugging systems.
///
/// LLDB instantiates TraceExporter plugins on-demand when a user requests
/// trace export (e.g., via the "thread trace export" command). The plugin
/// is selected by name via TraceExporter::FindPlugin(), which looks up the
/// appropriate exporter in the PluginManager.
///
/// Key methods to implement:
/// - Export functionality that reads from a TraceCursor and writes to the
///   target format
///
/// Implementation notes:
/// - Trace exporters are supposed to operate on an architecture-agnostic fashion,
///   as a TraceCursor, which feeds the data, hides the actual trace technology
///   being used
/// - The exporter should handle streaming large trace datasets efficiently
/// - Error handling is important as export operations can fail due to I/O issues
/// - Examples include exporters to Chrome's trace event format, CTF, or custom
///   analysis formats
class TraceExporter : public PluginInterface {
public:
  /// Create an instance of a trace exporter plugin given its name.
  ///
  /// \param[in] plugin_Name
  ///     Plug-in name to search.
  ///
  /// \return
  ///     A \a TraceExporterUP instance, or an \a llvm::Error if the plug-in
  ///     name doesn't match any registered plug-ins.
  static llvm::Expected<lldb::TraceExporterUP>
  FindPlugin(llvm::StringRef plugin_name);
};

} // namespace lldb_private

#endif // LLDB_TARGET_TRACEEXPORTER_H
