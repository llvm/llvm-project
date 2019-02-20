//===-- HostThreadPosix.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_posix_HostThreadPosix_h_
#define lldb_Host_posix_HostThreadPosix_h_

#include "lldb/Host/HostNativeThreadBase.h"

namespace lldb_private {

class HostThreadPosix : public HostNativeThreadBase {
  DISALLOW_COPY_AND_ASSIGN(HostThreadPosix);

public:
  HostThreadPosix();
  HostThreadPosix(lldb::thread_t thread);
  ~HostThreadPosix() override;

  Status Join(lldb::thread_result_t *result) override;
  Status Cancel() override;

  Status Detach();
};

} // namespace lldb_private

#endif // lldb_Host_posix_HostThreadPosix_h_
