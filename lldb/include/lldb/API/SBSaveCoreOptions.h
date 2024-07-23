//===-- SBSaveCoreOptions.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBSAVECOREOPTIONS_H
#define LLDB_API_SBSAVECOREOPTIONS_H

#include "lldb/API/SBDefines.h"

namespace lldb {

class LLDB_API SBSaveCoreOptions {
public:
  SBSaveCoreOptions();
  SBSaveCoreOptions(const lldb::SBSaveCoreOptions &rhs);
  ~SBSaveCoreOptions() = default;

  const SBSaveCoreOptions &operator=(const lldb::SBSaveCoreOptions &rhs);

  /// Set the plugin name. Supplying null or empty string will reset
  /// the option.
  ///
  /// \param plugin Name of the object file plugin.
  SBError SetPluginName(const char *plugin);

  /// Get the Core dump plugin name, if set.
  ///
  /// \return The name of the plugin, or null if not set.
  const char *GetPluginName() const;

  /// Set the Core dump style.
  ///
  /// \param style The style of the core dump.
  void SetStyle(lldb::SaveCoreStyle style);

  /// Get the Core dump style, if set.
  ///
  /// \return The core dump style, or undefined if not set.
  lldb::SaveCoreStyle GetStyle() const;

  /// Set the output file path
  ///
  /// \param output_file a
  /// \class SBFileSpec object that describes the output file.
  void SetOutputFile(SBFileSpec output_file);

  /// Get the output file spec
  ///
  /// \return The output file spec.
  SBFileSpec GetOutputFile() const;

  /// Add a thread to save in the core file.
  ///
  /// \param thread_id The thread ID to save.
  void AddThread(lldb::tid_t thread_id);

  /// Remove a thread from the list of threads to save.
  ///
  /// \param thread_id The thread ID to remove.
  /// \return True if the thread was removed, false if it was not in the list.
  bool RemoveThread(lldb::tid_t thread_id);

  /// Get the number of threads to save. If this list is empty all threads will
  /// be saved.
  ///
  /// \return The number of threads to save.
  uint32_t GetNumThreads() const;

  /// Get the thread ID at the given index.
  ///
  /// \param[in] index The index of the thread ID to get.
  /// \return The thread ID at the given index, or an error
  /// if there is no thread at the index.
  lldb::tid_t GetThreadAtIndex(uint32_t index, SBError &error) const;

  /// Reset all options.
  void Clear();

protected:
  friend class SBProcess;
  lldb_private::SaveCoreOptions &ref() const;

private:
  std::unique_ptr<lldb_private::SaveCoreOptions> m_opaque_up;
}; // SBSaveCoreOptions
} // namespace lldb

#endif // LLDB_API_SBSAVECOREOPTIONS_H
