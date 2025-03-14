//===-- SBLock.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBLock.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Instrumentation.h"
#include "lldb/lldb-forward.h"

#include <memory>
#include <mutex>

using namespace lldb;
using namespace lldb_private;

#ifndef SWIG

SBLock::SBLock(std::recursive_mutex &mutex, lldb::TargetSP target_sp)
    : m_opaque_up(std::make_unique<TargetAPILock>(mutex, target_sp)) {
  LLDB_INSTRUMENT_VA(this);
}

SBLock::~SBLock() { LLDB_INSTRUMENT_VA(this); }

bool SBLock::IsValid() const {
  LLDB_INSTRUMENT_VA(this);

  return static_cast<bool>(m_opaque_up);
}

#endif
