//===-- SBReproducer.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBREPRODUCER_H
#define LLDB_API_SBREPRODUCER_H

#include "lldb/API/SBDefines.h"

namespace lldb {

/// The SBReproducer class is special because it bootstraps the capture and
/// replay of SB API calls. As a result we cannot rely on any other SB objects
/// in the interface or implementation of this class.
class LLDB_API SBReproducer {
public:
  static const char *Capture();
  static const char *Capture(const char *path);
  static const char *Replay(const char *path);
};

} // namespace lldb

#endif
