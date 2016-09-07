//===-- ThisThread.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_ThisThread_h_
#define lldb_Host_ThisThread_h_

#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm {
template <class T> class SmallVectorImpl;
}

namespace lldb_private {

class ThisThread {
private:
  ThisThread();

public:
  // ThisThread common functions.
  static void SetName(llvm::StringRef name, int max_length);

  // ThisThread platform-specific functions.
  static void SetName(llvm::StringRef name);
  static void GetName(llvm::SmallVectorImpl<char> &name);
};
}

#endif
