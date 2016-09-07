//===-- AbstractSocket.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AbstractSocket_h_
#define liblldb_AbstractSocket_h_

#include "lldb/Host/posix/DomainSocket.h"

namespace lldb_private {
class AbstractSocket : public DomainSocket {
public:
  AbstractSocket(bool child_processes_inherit, Error &error);

protected:
  size_t GetNameOffset() const override;
  void DeleteSocketFile(llvm::StringRef name) override;
};
}

#endif // ifndef liblldb_AbstractSocket_h_
