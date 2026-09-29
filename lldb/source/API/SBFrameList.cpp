//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFrameList.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBThread.h"
#include "lldb/Target/StackFrameList.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Instrumentation.h"

using namespace lldb;
using namespace lldb_private;

SBFrameList::SBFrameList() : m_opaque_sp() { LLDB_INSTRUMENT_VA(this); }

SBFrameList::SBFrameList(const SBFrameList &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_INSTRUMENT_VA(this, rhs);
}

SBFrameList::~SBFrameList() = default;

const SBFrameList &SBFrameList::operator=(const SBFrameList &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    m_opaque_sp = rhs.m_opaque_sp;
  return *this;
}

SBFrameList::SBFrameList(const lldb::StackFrameListSP &frame_list_sp)
    : m_opaque_sp(frame_list_sp) {}

void SBFrameList::SetFrameList(const lldb::StackFrameListSP &frame_list_sp) {
  m_opaque_sp = frame_list_sp;
}

SBFrameList::operator bool() const {
  LLDB_INSTRUMENT_VA(this);

  return m_opaque_sp.get() != nullptr;
}

bool SBFrameList::IsValid() const {
  LLDB_INSTRUMENT_VA(this);
  return this->operator bool();
}

uint32_t SBFrameList::GetSize() const {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_sp)
    return m_opaque_sp->GetNumFrames();
  return 0;
}

SBFrame SBFrameList::GetFrameAtIndex(uint32_t idx) const {
  LLDB_INSTRUMENT_VA(this, idx);

  SBFrame sb_frame;
  if (m_opaque_sp)
    sb_frame.SetFrameSP(m_opaque_sp->GetFrameAtIndex(idx));
  return sb_frame;
}

SBThread SBFrameList::GetThread() const {
  LLDB_INSTRUMENT_VA(this);

  SBThread sb_thread;
  if (m_opaque_sp)
    sb_thread.SetThread(m_opaque_sp->GetThread().shared_from_this());
  return sb_thread;
}

void SBFrameList::Clear() {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_sp)
    m_opaque_sp->Clear();
}

bool SBFrameList::GetDescription(SBStream &description) const {
  LLDB_INSTRUMENT_VA(this, description);

  if (!m_opaque_sp)
    return false;

  Stream &strm = description.ref();
  m_opaque_sp->Dump(&strm);
  return true;
}
