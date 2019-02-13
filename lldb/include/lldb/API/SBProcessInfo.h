//===-- SBProcessInfo.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBProcessInfo_h_
#define LLDB_SBProcessInfo_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class LLDB_API SBProcessInfo {
public:
  SBProcessInfo();
  SBProcessInfo(const SBProcessInfo &rhs);

  ~SBProcessInfo();

  SBProcessInfo &operator=(const SBProcessInfo &rhs);

  bool IsValid() const;

  const char *GetName();

  SBFileSpec GetExecutableFile();

  lldb::pid_t GetProcessID();

  uint32_t GetUserID();

  uint32_t GetGroupID();

  bool UserIDIsValid();

  bool GroupIDIsValid();

  uint32_t GetEffectiveUserID();

  uint32_t GetEffectiveGroupID();

  bool EffectiveUserIDIsValid();

  bool EffectiveGroupIDIsValid();

  lldb::pid_t GetParentProcessID();

private:
  friend class SBProcess;

  lldb_private::ProcessInstanceInfo &ref();

  void SetProcessInfo(const lldb_private::ProcessInstanceInfo &proc_info_ref);

  std::unique_ptr<lldb_private::ProcessInstanceInfo> m_opaque_up;
};

} // namespace lldb

#endif // LLDB_SBProcessInfo_h_
