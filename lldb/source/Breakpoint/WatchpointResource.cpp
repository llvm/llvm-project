//===-- WatchpointResource.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/WatchpointResource.h"

using namespace lldb;
using namespace lldb_private;

WatchpointResource::WatchpointResource(lldb::addr_t addr, size_t size,
                                       bool read, bool write)
    : m_id(LLDB_INVALID_WATCHPOINT_RESOURCE_ID), m_addr(addr), m_size(size),
      m_watch_read(read), m_watch_write(write) {}

WatchpointResource::~WatchpointResource() {
  std::lock_guard<std::recursive_mutex> guard(m_owners_mutex);
  m_owners.Clear();
}

void WatchpointResource::GetMemoryRange(lldb::addr_t &addr,
                                        size_t &size) const {
  addr = m_addr;
  size = m_size;
}

addr_t WatchpointResource::GetAddress() const { return m_addr; }

size_t WatchpointResource::GetByteSize() const { return m_size; }

void WatchpointResource::GetType(bool &read, bool &write) const {
  read = m_watch_read;
  write = m_watch_write;
}

void WatchpointResource::SetType(bool read, bool write) {
  m_watch_read = read;
  m_watch_write = write;
}

wp_resource_id_t WatchpointResource::GetID() const { return m_id; }

void WatchpointResource::SetID(wp_resource_id_t id) { m_id = id; }

bool WatchpointResource::Contains(addr_t addr) {
  if (addr >= m_addr && addr < m_addr + m_size)
    return true;
  return false;
}

void WatchpointResource::AddOwner(const WatchpointSP &wp_sp) {
  std::lock_guard<std::recursive_mutex> guard(m_owners_mutex);
  m_owners.Add(wp_sp);
}

void WatchpointResource::RemoveOwner(WatchpointSP &wp_sp) {
  std::lock_guard<std::recursive_mutex> guard(m_owners_mutex);
  m_owners.Remove(wp_sp);
}

size_t WatchpointResource::GetNumberOfOwners() {
  std::lock_guard<std::recursive_mutex> guard(m_owners_mutex);
  return m_owners.GetSize();
}

bool WatchpointResource::OwnersContains(WatchpointSP &wp_sp) {
  std::lock_guard<std::recursive_mutex> guard(m_owners_mutex);
  return m_owners.Contains(wp_sp);
}

bool WatchpointResource::OwnersContains(const Watchpoint *wp) {
  std::lock_guard<std::recursive_mutex> guard(m_owners_mutex);
  return m_owners.Contains(wp);
}

WatchpointSP WatchpointResource::GetOwnerAtIndex(size_t idx) {
  std::lock_guard<std::recursive_mutex> guard(m_owners_mutex);
  assert(idx < m_owners.GetSize());
  if (idx >= m_owners.GetSize())
    return {};

  return m_owners.GetByIndex(idx);
}

size_t
WatchpointResource::CopyOwnersList(WatchpointCollection &out_collection) {
  std::lock_guard<std::recursive_mutex> guard(m_owners_mutex);
  const size_t size = m_owners.GetSize();
  for (size_t i = 0; i < size; ++i) {
    out_collection.Add(m_owners.GetByIndex(i));
  }
  return out_collection.GetSize();
}
