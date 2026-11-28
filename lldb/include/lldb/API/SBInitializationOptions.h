//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBINITIALIZATIONOPTIONS_H
#define LLDB_API_SBINITIALIZATIONOPTIONS_H

#include "lldb/API/SBDefines.h"

namespace lldb {

/// Options for SBDebugger::Initialize. They apply to the whole process.
class LLDB_API SBInitializationOptions {
public:
  SBInitializationOptions();
  SBInitializationOptions(const lldb::SBInitializationOptions &rhs);
  ~SBInitializationOptions();

  const SBInitializationOptions &
  operator=(const lldb::SBInitializationOptions &rhs);

protected:
  friend class SBDebugger;

  const lldb_private::InitializationOptions &ref() const;

private:
  std::unique_ptr<lldb_private::InitializationOptions> m_opaque_up;
};

} // namespace lldb

#endif // LLDB_API_SBINITIALIZATIONOPTIONS_H
