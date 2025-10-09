//===-- SBFrameList.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFrameList.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBStream.h"
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

void SBFrameList::SetOpaque(const lldb::StackFrameListSP &frame_list_sp) {
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

void SBFrameList::Clear() {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_sp)
    m_opaque_sp->Clear();
}

void SBFrameList::Append(const SBFrame &frame) {
  LLDB_INSTRUMENT_VA(this, frame);

  // Note: StackFrameList doesn't have an Append method, so this is a no-op
  // This method is kept for API consistency with other SB*List classes
}

void SBFrameList::Append(const SBFrameList &frame_list) {
  LLDB_INSTRUMENT_VA(this, frame_list);

  // Note: StackFrameList doesn't have an Append method, so this is a no-op
  // This method is kept for API consistency with other SB*List classes
}

bool SBFrameList::GetDescription(SBStream &description) const {
  LLDB_INSTRUMENT_VA(this, description);

  Stream &strm = description.ref();
  if (m_opaque_sp) {
    m_opaque_sp->Dump(&strm);
    return true;
  }
  return false;
}
