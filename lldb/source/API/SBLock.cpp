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

SBLock::SBLock() { LLDB_INSTRUMENT_VA(this); }

SBLock::SBLock(SBLock &&rhs) : m_opaque_up(std::move(rhs.m_opaque_up)) {
  LLDB_INSTRUMENT_VA(this);
}

SBLock &SBLock::operator=(SBLock &&rhs) {
  LLDB_INSTRUMENT_VA(this);

  m_opaque_up = std::move(rhs.m_opaque_up);
  return *this;
}

SBLock::SBLock(lldb::TargetSP target_sp)
    : m_opaque_up(
          std::make_unique<APILock>(std::shared_ptr<std::recursive_mutex>(
              target_sp, &target_sp->GetAPIMutex()))) {
  LLDB_INSTRUMENT_VA(this, target_sp);
}

SBLock::~SBLock() { LLDB_INSTRUMENT_VA(this); }

bool SBLock::IsValid() const {
  LLDB_INSTRUMENT_VA(this);

  return static_cast<bool>(m_opaque_up) && static_cast<bool>(*m_opaque_up);
}

void SBLock::Lock() const {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_up)
    m_opaque_up->Lock();
}

void SBLock::Unlock() const {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_up)
    m_opaque_up->Unlock();
}
