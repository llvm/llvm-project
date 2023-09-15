//===-- SBWatchpointOptions.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBWatchpointOptions.h"
#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Utility/Instrumentation.h"

#include "Utils.h"

using namespace lldb;
using namespace lldb_private;

class WatchpointOptionsImpl {
public:
  bool m_read = false;
  bool m_write = false;
  bool m_modify = false;
};


SBWatchpointOptions::SBWatchpointOptions()
    : m_opaque_up(new WatchpointOptionsImpl()) {
  LLDB_INSTRUMENT_VA(this);
}

SBWatchpointOptions::SBWatchpointOptions(const SBWatchpointOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

const SBWatchpointOptions &
SBWatchpointOptions::operator=(const SBWatchpointOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

SBWatchpointOptions::~SBWatchpointOptions() = default;

void SBWatchpointOptions::SetWatchpointTypeRead(bool read) {
  m_opaque_up->m_read = read;
}
bool SBWatchpointOptions::GetWatchpointTypeRead() const {
  return m_opaque_up->m_read;
}

void SBWatchpointOptions::SetWatchpointTypeWrite(bool write) {
  m_opaque_up->m_write = write;
}
bool SBWatchpointOptions::GetWatchpointTypeWrite() const {
  return m_opaque_up->m_write;
}

void SBWatchpointOptions::SetWatchpointTypeModify(bool modify) {
  m_opaque_up->m_modify = modify;
}
bool SBWatchpointOptions::GetWatchpointTypeModify() const {
  return m_opaque_up->m_modify;
}
