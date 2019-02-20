//===-- LockFilePosix.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_posix_LockFilePosix_h_
#define liblldb_Host_posix_LockFilePosix_h_

#include "lldb/Host/LockFileBase.h"

namespace lldb_private {

class LockFilePosix : public LockFileBase {
public:
  explicit LockFilePosix(int fd);
  ~LockFilePosix() override;

protected:
  Status DoWriteLock(const uint64_t start, const uint64_t len) override;

  Status DoTryWriteLock(const uint64_t start, const uint64_t len) override;

  Status DoReadLock(const uint64_t start, const uint64_t len) override;

  Status DoTryReadLock(const uint64_t start, const uint64_t len) override;

  Status DoUnlock() override;
};

} // namespace lldb_private

#endif // liblldb_Host_posix_LockFilePosix_h_
