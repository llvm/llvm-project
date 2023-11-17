//===-- WatchpointResourceList.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/WatchpointResourceList.h"
#include "lldb/Breakpoint/WatchpointResource.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

WatchpointResourceList::WatchpointResourceList() : m_resources(), m_mutex() {}

WatchpointResourceList::~WatchpointResourceList() { Clear(); }

wp_resource_id_t
WatchpointResourceList::Add(const WatchpointResourceSP &wp_res_sp) {
  Log *log = GetLog(LLDBLog::Watchpoints);
  std::lock_guard<std::mutex> guard(m_mutex);
  LLDB_LOGF(log, "WatchpointResourceList::Add(addr 0x%" PRIx64 " size %zu)",
            wp_res_sp->GetLoadAddress(), wp_res_sp->GetByteSize());

  m_resources.push_back(wp_res_sp);
  return wp_res_sp->GetID();
}

bool WatchpointResourceList::Remove(wp_resource_id_t id) {
  std::lock_guard<std::mutex> guard(m_mutex);
  Log *log = GetLog(LLDBLog::Watchpoints);
  for (collection::iterator pos = m_resources.begin(); pos != m_resources.end();
       ++pos) {
    if ((*pos)->GetID() == id) {
      LLDB_LOGF(log,
                "WatchpointResourceList::Remove(addr 0x%" PRIx64 " size %zu)",
                (*pos)->GetLoadAddress(), (*pos)->GetByteSize());
      m_resources.erase(pos);
      return true;
    }
  }
  return false;
}

bool WatchpointResourceList::RemoveByAddress(addr_t addr) {
  std::lock_guard<std::mutex> guard(m_mutex);
  Log *log = GetLog(LLDBLog::Watchpoints);
  for (collection::iterator pos = m_resources.begin(); pos != m_resources.end();
       ++pos) {
    if ((*pos)->Contains(addr)) {
      LLDB_LOGF(log,
                "WatchpointResourceList::RemoveByAddress(addr 0x%" PRIx64
                " size %zu)",
                (*pos)->GetLoadAddress(), (*pos)->GetByteSize());
      m_resources.erase(pos);
      return true;
    }
  }
  return false;
}

WatchpointResourceSP WatchpointResourceList::FindByAddress(addr_t addr) {
  std::lock_guard<std::mutex> guard(m_mutex);
  for (WatchpointResourceSP wp_res_sp : m_resources)
    if (wp_res_sp->Contains(addr))
      return wp_res_sp;
  return {};
}

WatchpointResourceSP
WatchpointResourceList::FindByWatchpointSP(WatchpointSP &wp_sp) {
  return FindByWatchpoint(wp_sp.get());
}

WatchpointResourceSP
WatchpointResourceList::FindByWatchpoint(const Watchpoint *wp) {
  std::lock_guard<std::mutex> guard(m_mutex);
  for (WatchpointResourceSP wp_res_sp : m_resources)
    if (wp_res_sp->ConstituentsContains(wp))
      return wp_res_sp;
  return {};
}

WatchpointResourceSP WatchpointResourceList::FindByID(wp_resource_id_t id) {
  std::lock_guard<std::mutex> guard(m_mutex);
  for (WatchpointResourceSP wp_res_sp : m_resources)
    if (wp_res_sp->GetID() == id)
      return wp_res_sp;
  return {};
}

uint32_t WatchpointResourceList::GetSize() {
  std::lock_guard<std::mutex> guard(m_mutex);
  return m_resources.size();
}

lldb::WatchpointResourceSP
WatchpointResourceList::GetResourceAtIndex(uint32_t idx) {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (idx < m_resources.size())
    return m_resources[idx];

  return {};
}

void WatchpointResourceList::Clear() {
  std::lock_guard<std::mutex> guard(m_mutex);
  m_resources.clear();
}

std::mutex &WatchpointResourceList::GetMutex() { return m_mutex; }
