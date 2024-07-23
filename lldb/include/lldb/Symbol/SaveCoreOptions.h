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

#include <optional>
#include <set>
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

  void AddThread(lldb::tid_t tid);
  bool RemoveThread(lldb::tid_t tid);
  size_t GetNumThreads() const;
  int64_t GetThreadAtIndex(size_t index) const;
  bool ShouldSaveThread(lldb::tid_t tid) const;

  Status EnsureValidConfiguration() const;

  void Clear();

private:
  std::optional<std::string> m_plugin_name;
  std::optional<lldb_private::FileSpec> m_file;
  std::optional<lldb::SaveCoreStyle> m_style;
  std::set<lldb::tid_t> m_threads_to_save;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_SaveCoreOPTIONS_H
