//===-- SBMutex.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBMutex.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Instrumentation.h"
#include "lldb/lldb-forward.h"
#include <memory>
#include <mutex>

using namespace lldb;
using namespace lldb_private;

SBMutex::SBMutex() : m_opaque_sp(std::make_shared<std::recursive_mutex>()) {
  LLDB_INSTRUMENT_VA(this);
}

SBMutex::SBMutex(const SBMutex &rhs) : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_INSTRUMENT_VA(this);
}

const SBMutex &SBMutex::operator=(const SBMutex &rhs) {
  LLDB_INSTRUMENT_VA(this);

  m_opaque_sp = rhs.m_opaque_sp;
  return *this;
}

SBMutex::SBMutex(lldb::TargetSP target_sp)
    : m_opaque_sp(std::shared_ptr<std::recursive_mutex>(
          target_sp, &target_sp->GetAPIMutex())) {
  LLDB_INSTRUMENT_VA(this, target_sp);
}

SBMutex::~SBMutex() { LLDB_INSTRUMENT_VA(this); }

bool SBMutex::IsValid() const {
  LLDB_INSTRUMENT_VA(this);

  return static_cast<bool>(m_opaque_sp);
}

void SBMutex::lock() const {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_sp)
    m_opaque_sp->lock();
}

void SBMutex::unlock() const {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_sp)
    m_opaque_sp->unlock();
}
