//===-- Uio.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_linux_Uio_h_
#define liblldb_Host_linux_Uio_h_

#include "lldb/Host/Config.h"
#include <sys/uio.h>

// We shall provide our own implementation of process_vm_readv if it is not
// present
#if !HAVE_PROCESS_VM_READV
ssize_t process_vm_readv(::pid_t pid, const struct iovec *local_iov,
                         unsigned long liovcnt, const struct iovec *remote_iov,
                         unsigned long riovcnt, unsigned long flags);
#endif

#endif // liblldb_Host_linux_Uio_h_
