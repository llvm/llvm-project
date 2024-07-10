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
    CoreDumpOptions(const lldb_private::FileSpec &fspec) :
      m_core_dump_file(std::move(fspec)) {};
    ~CoreDumpOptions() = default;


  void SetCoreDumpPluginName(llvm::StringRef name);
  std::optional<llvm::StringRef> GetCoreDumpPluginName() const;

  void SetCoreDumpStyle(lldb::SaveCoreStyle style);
  lldb::SaveCoreStyle GetCoreDumpStyle() const;

  const lldb_private::FileSpec& GetOutputFile() const;

private:
  std::optional<std::string> m_core_dump_plugin_name;
  const lldb_private::FileSpec m_core_dump_file;
  lldb::SaveCoreStyle m_core_dump_style = lldb::eSaveCoreUnspecified;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_COREDUMPOPTIONS_H
