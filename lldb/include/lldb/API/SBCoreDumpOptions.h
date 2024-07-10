//===-- SBCoreDumpOptions.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBCOREDUMPOPTIONS_H
#define LLDB_API_SBCOREDUMPOPTIONS_H

#include "lldb/API/SBDefines.h"
#include "lldb/Symbol/CoreDumpOptions.h"

namespace lldb {

class LLDB_API SBCoreDumpOptions {
public:
  SBCoreDumpOptions(const char *filePath);
  SBCoreDumpOptions(const lldb::SBCoreDumpOptions &rhs);
  ~SBCoreDumpOptions() = default;

  const SBCoreDumpOptions &operator=(const lldb::SBCoreDumpOptions &rhs);

  /// Set the Core dump plugin name.
  ///
  /// \param plugin Name of the object file plugin.
  void SetCoreDumpPluginName(const char *plugin);

  /// Get the Core dump plugin name, if set.
  ///
  /// \return The name of the plugin, or nullopt if not set.
  const std::optional<const char *> GetCoreDumpPluginName() const;

  /// Set the Core dump style.
  ///
  /// \param style The style of the core dump.
  void SetCoreDumpStyle(lldb::SaveCoreStyle style);

  /// Get the Core dump style, if set.
  ///
  /// \return The core dump style, or nullopt if not set.
  const std::optional<lldb::SaveCoreStyle> GetCoreDumpStyle() const;

  /// Get the output file path
  ///
  /// \return The output file path.
  const char *GetOutputFile() const;

protected:
  friend class SBProcess;
  lldb_private::CoreDumpOptions &Ref() const;

private:
  std::unique_ptr<lldb_private::CoreDumpOptions> m_opaque_up;
}; // SBCoreDumpOptions
} // namespace lldb

#endif // LLDB_API_SBCOREDUMPOPTIONS_H
