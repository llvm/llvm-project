//===-- CoreDumpOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTFILE_COREDUMPOPTIONS_H
#define LLDB_SOURCE_PLUGINS_OBJECTFILE_COREDUMPOPTIONS_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"

#include <optional>
#include <string>

namespace lldb_private {

class CoreDumpOptions {
public:
  CoreDumpOptions() {};
  ~CoreDumpOptions() = default;

  void SetCoreDumpPluginName(const char * name);
  std::optional<std::string> GetCoreDumpPluginName() const;

  void SetCoreDumpStyle(lldb::SaveCoreStyle style);
  lldb::SaveCoreStyle GetCoreDumpStyle() const;

  void SetOutputFile(lldb_private::FileSpec file);
  const std::optional<lldb_private::FileSpec> GetOutputFile() const;

  void Clear();

private:
  std::optional<std::string> m_plugin_name;
  std::optional<lldb_private::FileSpec> m_file;
  std::optional<lldb::SaveCoreStyle> m_style;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_COREDUMPOPTIONS_H
