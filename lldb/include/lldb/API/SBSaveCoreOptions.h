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
#include "lldb/API/SBError.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBThread.h"

namespace lldb {

class LLDB_API SBSaveCoreOptions {
public:
  SBSaveCoreOptions();
  SBSaveCoreOptions(const lldb::SBSaveCoreOptions &rhs);
  ~SBSaveCoreOptions();

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

  /// Set the process to save, or unset if supplied with a default constructed
  /// process.
  ///
  /// \param process The process to save.
  /// \return Success if process was set, otherwise an error
  /// \note This will clear all process specific options if a different process
  /// is specified than the current set process, either explicitly from this
  /// api, or implicitly from any function that requires a process.
  SBError SetProcess(lldb::SBProcess process);

  /// Add a thread to save in the core file.
  ///
  /// \param thread The thread to save.
  /// \note This will set the process if it is not already set, or return
  /// and error if the SBThread is not from the set process.
  SBError AddThread(lldb::SBThread thread);

  /// Remove a thread from the list of threads to save.
  ///
  /// \param thread The thread to remove.
  /// \return True if the thread was removed, false if it was not in the list.
  bool RemoveThread(lldb::SBThread thread);

  /// Add a memory region to save in the core file.
  ///
  /// \param region The memory region to save.
  /// \returns An empty SBError upon success, or an error if the region is
  /// invalid.
  /// \note Ranges that overlapped will be unioned into a single region, this
  /// also supercedes stack minification. Specifying full regions and a
  /// non-custom core style will include the specified regions and union them
  /// with all style specific regions.
  SBError AddMemoryRegionToSave(const SBMemoryRegionInfo &region);

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
