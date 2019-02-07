//===-- SBReproducer.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBREPRODUCER_H
#define LLDB_API_SBREPRODUCER_H

#include "lldb/lldb-defines.h"

namespace lldb {

class LLDB_API SBReproducer {
public:
  static bool Replay();
};

} // namespace lldb

#endif
