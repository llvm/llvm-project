//===-- SBCoreDumpOptions.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBCOREDUMPOPTIONS_H
#define LLDB_API_SBCOREDUMPOPTIONS_H

namespace lldb {

class LLDB_API SBCoreDumpOptions {
public:
  SBCoreDumpOptions() {};
  SBStatisticsOptions(const lldb::SBCoreDumpOptions &rhs);
  ~SBExpressionOptions() = default;

  const SBStatisticsOptions &operator=(const lldb::SBStatisticsOptions &rhs);

  void SetCoreDumpPluginName(const char* plugin);
  const char* GetCoreDumpPluginName();

  void SetCoreDumpStyle(const char* style);
  const char* GetCoreDumpStyle();

  void SetOutputFilePath(const char* path);
  const char* GetOutputFilePath();

private:
  std::unique_ptr<lldb_private::SBCoreDumpOptions> m_opaque_up;
}; // SBCoreDumpOptions

}; // namespace lldb

#endif // LLDB_API_SBCOREDUMPOPTIONS_H
