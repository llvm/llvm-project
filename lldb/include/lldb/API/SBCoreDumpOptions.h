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
  SBCoreDumpOptions(const char* filePath);
  SBCoreDumpOptions(const lldb::SBCoreDumpOptions &rhs);
  ~SBCoreDumpOptions() = default;

  const SBCoreDumpOptions &operator=(const lldb::SBCoreDumpOptions &rhs);

  void SetCoreDumpPluginName(const char* plugin);
  const std::optional<const char *> GetCoreDumpPluginName() const;

  void SetCoreDumpStyle(lldb::SaveCoreStyle style);
  const std::optional<lldb::SaveCoreStyle> GetCoreDumpStyle() const;

  const char * GetOutputFile() const;

protected:
  friend class SBProcess;
  lldb_private::CoreDumpOptions &Ref() const;

private:
  std::unique_ptr<lldb_private::CoreDumpOptions> m_opaque_up;
}; // SBCoreDumpOptions
} // namespace lldb

#endif // LLDB_API_SBCOREDUMPOPTIONS_H
