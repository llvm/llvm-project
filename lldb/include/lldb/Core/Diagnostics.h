//===-- Diagnostics.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_DIAGNOSTICS_H
#define LLDB_CORE_DIAGNOSTICS_H

#include "lldb/Core/UserSettingsController.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
namespace json {
class Value;
} // namespace json
} // namespace llvm

namespace lldb_private {

class Debugger;
class ExecutionContext;

/// The global diagnostics settings, exposed under `diagnostics` in the settings
/// hierarchy.
class DiagnosticsProperties : public Properties {
public:
  DiagnosticsProperties();

  bool GetCollectBinaries() const;
  bool SetCollectBinaries(bool collect);
};

/// Diagnostics maintain an always-on, in-memory log of recent diagnostic
/// messages that can be written out to help investigate bugs and troubleshoot
/// issues.
class Diagnostics {
public:
  Diagnostics();
  ~Diagnostics();

  /// The bundle directory and the files written into it, recorded as each one
  /// is created so a file that could not be written is simply absent.
  struct Attachments {
    std::string directory;
    std::vector<std::string> files;
  };

  /// The state a triager needs to make sense of a bug report. The full payload
  /// is written into the bundle directory. These scalars are carried in the
  /// terminal output and the report body rather than as redundant files. More
  /// fields are expected to accrue over time.
  struct Report {
    std::string version;
    std::string os;
    std::string invocation;
    Attachments attachments;
  };

  /// Write the in-memory diagnostic log into the given directory.
  llvm::Error Create(const FileSpec &dir);

  /// Collect a full diagnostics bundle into \p dir and return its report.
  ///
  /// Writes the always-on log, the debugger's file-backed logs, statistics,
  /// and a snapshot of the commands a triager runs first. When the
  /// `collect-binaries` setting is enabled it also copies the executable, its
  /// symbol file, and the core file. Collection is best-effort: a failure to
  /// produce one artifact never aborts the rest, so a partial bundle is always
  /// better than none.
  llvm::Expected<Report> Collect(Debugger &debugger,
                                 const ExecutionContext &exe_ctx,
                                 const FileSpec &dir);

  /// Write the diagnostic log into a directory and print a message to the given
  /// output stream.
  /// @{
  bool Dump(llvm::raw_ostream &stream);
  bool Dump(llvm::raw_ostream &stream, const FileSpec &dir);
  /// @}

  /// Record a diagnostic message into the always-on, in-memory log.
  void Record(llvm::StringRef message);

  /// Supplies an artifact's contents on demand. Subsystems register a provider
  /// so Core need not depend on them. Each runs when a bundle is collected.
  using ArtifactProvider = std::function<std::string()>;
  using ArtifactProviderID = uint64_t;

  /// Register \p provider to contribute file \p name. Returns an id for
  /// RemoveArtifactProvider. Thread-safe.
  ArtifactProviderID AddArtifactProvider(std::string name,
                                         ArtifactProvider provider);

  /// Unregister a provider. Thread-safe.
  void RemoveArtifactProvider(ArtifactProviderID id);

  static Diagnostics &Instance();

  static DiagnosticsProperties &GetGlobalProperties();

  static bool Enabled();
  static void Initialize();
  static void Terminate();

  /// Create a unique diagnostic directory.
  static llvm::Expected<FileSpec> CreateUniqueDirectory();

private:
  static std::optional<Diagnostics> &InstanceImpl();

  llvm::Error DumpDiangosticsLog(const FileSpec &dir) const;

  /// Collect the individual parts of the bundle into \p dir, appending the name
  /// of each file to \p files as it is written.
  /// @{
  void CollectLogs(Debugger &debugger, const FileSpec &dir,
                   std::vector<std::string> &files);
  static void CollectStatistics(Debugger &debugger,
                                const ExecutionContext &exe_ctx,
                                const FileSpec &dir,
                                std::vector<std::string> &files);
  static void CollectCommands(Debugger &debugger,
                              const ExecutionContext &exe_ctx,
                              const FileSpec &dir,
                              std::vector<std::string> &files);
  static void CollectBinaries(const ExecutionContext &exe_ctx,
                              const FileSpec &dir,
                              std::vector<std::string> &files);
  void CollectArtifactProviders(const FileSpec &dir,
                                std::vector<std::string> &files);
  /// @}

  /// Scalars carried in the report rather than written as files.
  /// @{
  static std::string GetHostDescription(const ExecutionContext &exe_ctx);
  static std::string GetInvocation();
  /// @}

  RotatingLogHandler m_log_handler;

  struct ArtifactProviderEntry {
    ArtifactProviderID id;
    std::string name;
    ArtifactProvider provider;
  };

  /// Registered artifact providers, guarded by the mutex.
  /// @{
  ArtifactProviderID m_next_artifact_provider_id = 0;
  std::vector<ArtifactProviderEntry> m_artifact_providers;
  std::mutex m_artifact_providers_mutex;
  /// @}
};

/// Render a diagnostics report as JSON, for `diagnostics dump`'s terminal
/// output.
llvm::json::Value toJSON(const Diagnostics::Report &report);

} // namespace lldb_private

#endif
