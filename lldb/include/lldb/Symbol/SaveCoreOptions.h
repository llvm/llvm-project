//===-- SaveCoreOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTFILE_SaveCoreOPTIONS_H
#define LLDB_SOURCE_PLUGINS_OBJECTFILE_SaveCoreOPTIONS_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"

#include <map>
#include <optional>
#include <string>

namespace lldb_private {

class SaveCoreOptions {
public:
  SaveCoreOptions(){};
  ~SaveCoreOptions() = default;

  lldb_private::Status SetPluginName(const char *name);
  std::optional<std::string> GetPluginName() const;

  void SetStyle(lldb::SaveCoreStyle style);
  lldb::SaveCoreStyle GetStyle() const;

  void SetOutputFile(lldb_private::FileSpec file);
  const std::optional<lldb_private::FileSpec> GetOutputFile() const;

  Status SetProcess(lldb::ProcessSP process_sp);

  Status AddThread(lldb_private::Thread *thread);
  bool RemoveThread(lldb_private::Thread *thread);
  bool ShouldSaveThread(lldb::tid_t tid) const;

  Status EnsureValidConfiguration(lldb::ProcessSP process_to_save) const;

  void Clear();

private:
  void ClearProcessSpecificData();

  std::optional<std::string> m_plugin_name;
  std::optional<lldb_private::FileSpec> m_file;
  std::optional<lldb::SaveCoreStyle> m_style;
  std::optional<lldb::ProcessSP> m_process_sp;
  std::map<lldb::tid_t, lldb::ThreadSP> m_threads_to_save;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_SaveCoreOPTIONS_H
