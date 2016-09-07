//===-- HostThreadFreeBSD.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_freebsd_HostThreadFreeBSD_h_
#define lldb_Host_freebsd_HostThreadFreeBSD_h_

#include "lldb/Host/posix/HostThreadPosix.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

class HostThreadFreeBSD : public HostThreadPosix {
public:
  HostThreadFreeBSD();
  HostThreadFreeBSD(lldb::thread_t thread);

  static void GetName(lldb::tid_t tid, llvm::SmallVectorImpl<char> &name);
};
}

#endif
