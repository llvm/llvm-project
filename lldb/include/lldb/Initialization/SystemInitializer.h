//===-- SystemInitializer.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INITIALIZATION_SYSTEM_INITIALIZER_H
#define LLDB_INITIALIZATION_SYSTEM_INITIALIZER_H

namespace lldb_private {
class SystemInitializer {
public:
  SystemInitializer();
  virtual ~SystemInitializer();

  virtual void Initialize() = 0;
  virtual void Terminate() = 0;
};
}

#endif
