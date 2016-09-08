//===-- LockFilePosix.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  Error DoWriteLock(const uint64_t start, const uint64_t len) override;

  Error DoTryWriteLock(const uint64_t start, const uint64_t len) override;

  Error DoReadLock(const uint64_t start, const uint64_t len) override;

  Error DoTryReadLock(const uint64_t start, const uint64_t len) override;

  Error DoUnlock() override;
};

} // namespace lldb_private

#endif // liblldb_Host_posix_LockFilePosix_h_
